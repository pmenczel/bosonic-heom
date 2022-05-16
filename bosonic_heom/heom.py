import numpy as np
from scipy.sparse import csr_matrix

from qutip.mesolve import mesolve

from qutip import expect, Qobj, sesolve
from qutip.operators import commutator, qeye
from qutip.solver import Options, Result
from qutip.states import enr_state_dictionaries
from qutip.superoperator import operator_to_vector, spre, spost, vector_to_operator

from . import util


class HEOM:
    r"""Represents the hierarchical equations of motion (HEOM).

    The HEOM are the equations of motion in an extended state space made up from multiple copies of
    the original state space, which contain auxiliary density operators (ADOs). Here, we construct
    the HEOM Liouvillian L which governs the time evolution, d_t rho_t = -i L_t rho_t, where rho_t
    is the extended state consisting of the system state and the ADOs.

    Note that the HEOM Liouvillian is time-dependent if the system Hamiltonian is time-dependent. We
    separate the system Hamiltonian into a static and a dynamic contribution,
        H_S = H_static + H_dynamic(t).
    The contribution of the thermal reservoirs is static and provided in the form of the influence
    phases of the reservoirs.

    In principle, there is an infinite number of ADOs. For numerical calculations, we introduce a
    cutoff and set all ADOs at deeper hierarchy levels to zero.

    In summary, that makes three parameters required to construct the Liouvillian:
        * static_hamiltonian: The (static part of the) system Hamiltonian
        * influence_phases: List of InfluencePhase objects
        * cutoff: Positive integer
    Once the (static part of the) Liouvillian is constructed, it is also available as a member
    variable (heom_liouvillian)."""

    def __init__(self, static_hamiltonian, influence_phases, cutoff):
        r"""Sets the parameters and constructs the HEOM Liouvillian."""

        self.static_hamiltonian = static_hamiltonian
        self.influence_phases = influence_phases
        self.cutoff = cutoff
        self.setup()

    def setup(self):
        r"""Constructs the HEOM Liouvillian. Call this function after altering any of the
        static_hamiltonian, influence_phases or cutoff parameters."""

        # Instead of indexing the ADOs by N_B multi-indices of dimension N_{f,1}, ..., N_{f,N_B}
        #   (where N_B is the number of reservoirs and N_{f,mu} is the number of frequencies in the
        #   correlation function of the mu-th reservoir), we will internally use a single
        #   multi-index of dimension N_{f,1} + ... + N_{f,N_B}. For each mode J, we now need to
        #   remember not only nu_J and T_J, but also Qx_J, the commutation superoperator
        #   with the bath coupling operator of the reservoir which that mode comes from.
        self._freqs = []
        self._thetas = []
        self._qxs = []
        for phase in self.influence_phases:
            for i in range(phase.num_terms):
                self._freqs.append(phase.freqs[i])
                self._thetas.append(phase.thetas[i])
                self._qxs.append(phase.coupling_superop)
        total_terms = len(self._freqs) # total number of frequencies of all reservoirs
                                       # = dimension of the single multi-index
        self._freqs = np.array(self._freqs) # comes handy later

        # For later purposes, we keep track of how many frequencies there were in each bath
        self._terms_per_bath = [phase.num_terms for phase in self.influence_phases]

        # We need to enumerate all multi-indices of dimension total_terms with absolute value less
        #   than or equal to cutoff. Luckily there is a function for that in QuTiP.
        self._num_ados, self._multi_to_index, self._index_to_multi =\
            enr_state_dictionaries([self.cutoff + 1] * total_terms, self.cutoff)

        # We are now ready to build the HEOM Liouvillian. The following superoperator is a
        # contribution which appears identically on every hierarchy level:
        # [H_S, ] - i * sum_(baths) c_delta * Qx**2
        # (Recall that we factored out a global "-i" in the definition of the HEOM Liouvillian.)
        constant_cont = spre(self.static_hamiltonian) - spost(self.static_hamiltonian) -\
            1j * sum([phase.c_delta * phase.coupling_superop**2 for phase in self.influence_phases])

        # We build up the HEOM Liouvillian level by level, saving all constructed terms in the
        # following list as we go. In order to convert superoperators acting on a single state
        # space into ones acting on the extended state space (consisting of self._num_ados copies),
        # we will use HEOM._extended_superoperator.
        terms = [HEOM._diag_superoperator(constant_cont, self._num_ados)]
        for n in range(self._num_ados): # pylint: disable=invalid-name
            multi = list(self._index_to_multi[n]) # multi is the actual multi-index, n is its
                                                  # position in our enumeration of multi-indices

            # ----- contribution coming from the present level -----
            dim = self.static_hamiltonian.shape[0]
            terms.append(HEOM._extended_superoperator(1j * sum(multi*self._freqs) * spre(qeye(dim)),
                                                      self._num_ados, n, n))

            for J in range(total_terms): # pylint: disable=invalid-name
                # ----- contribution from the next-deeper level -----
                if sum(multi) < self.cutoff:
                    # create the deeper multi-index, and find its position in the index enumeration
                    multi[J] += 1
                    n_next = self._multi_to_index[tuple(multi)]
                    multi[J] -= 1
                    terms.append(HEOM._extended_superoperator(-1j * self._qxs[J],
                                                              self._num_ados, n, n_next))

                # ----- contribution from the next-higher level -----
                if multi[J] > 0:
                    multi[J] -= 1
                    n_prev = self._multi_to_index[tuple(multi)]
                    multi[J] += 1
                    terms.append(HEOM._extended_superoperator(-1j * multi[J] * self._thetas[J],
                                                              self._num_ados, n, n_prev))

        # Done.
        self.heom_liouvillian = sum(terms)

    # pylint: disable=W0102, R0913
    def solve(self, rho0, tlist, e_ops=[], dynamic_hamiltonian=None, options=None, **kwargs):
        r"""Time evolution given an initial state.

        Parameters
        ----------
        rho0: Initial density matrix.
        tlist: List of times at which the state and expectation values will be returned.
        e_ops: List of system operators of which the expectation values will be evaluated.
            The list also may contain callback functions with the signature e(t, H(t), ados).
            Within these callback function, ados(..., [n_1, ..., n_Jk], ...) can then be used to
                obtain the ADO with the given multi-indices (one multi-index per reservoir).
            For example, bath_heat_current and system_heat_current yield callback functions which,
                when provided in the e_ops list, will calculate the bath (system) heat current into
                a specified reservoir at each time.
            Note that the ADOs here differ from the ones used e.g. in [Kato and Tanimura, J. Chem.
                Phys (2016)] by a phase: ADO (definition Tanimura) = (-i)^level * ADO (def. here),
                where "level" is the sum of the entries of all multi-indices.
        dynamic_hamiltonian: Callback function with signature dynamic_hamiltonian(t, **kwargs),
            which returns the dynamic Hamiltonian at time t.
        options: qutip.solver.Options to pass to the solver.
        kwargs: Additional arguments for the dynamical_hamiltonian function.

        Returns
        -------
        A qutip.Result containing the system states (if options.store_states is True), and the
        requested expectation values, at the times tlist."""

        if options is None:
            options = Options()

        # At the end of the day, the HEOM is just a linear differential equation; it can thus be
        # solved using sesolve.
        # --- Also you can be a Schrödinger equation, if you just believe in yourself! ---
        # However, we will create the Result ourselves. We therefore do not want sesolve to store
        # all the extended states:
        store_states = options.store_states
        options.store_states = False

        result = Result()
        result.solver = "heom"
        result.times = tlist
        result.num_expect = len(e_ops)
        result.expect = [[] for _ in range(result.num_expect)]
        if store_states:
            result.states = []
        else:
            result.states = None
        result.num_collapse = -1 # not applicable

        # Translate given initial state to the extended space
        extended_rho0 = HEOM._extended_operator(rho0, self._num_ados)

        # Handle dynamic part of Hamiltonian
        if dynamic_hamiltonian is None:
            # Just pass the constant HEOM Liouvillian to sesolve
            h_param = self.heom_liouvillian
        else:
            def h_param(time, _):
                result = self.heom_liouvillian
                h_dyn = dynamic_hamiltonian(time, **kwargs)
                result += HEOM._diag_superoperator(spre(h_dyn) - spost(h_dyn), self._num_ados)
                return result

        # Ready to do the integration. The following callback function will be called at every time
        # step in order to build up the Result object.
        def callback(time, extended_state):
            state = HEOM._extract_ado(extended_state, self._num_ados, 0)
            hamiltonian = self.static_hamiltonian
            if dynamic_hamiltonian is not None:
                hamiltonian += dynamic_hamiltonian(time, **kwargs)

            if store_states:
                result.states.append(state)
            for i, e_op in enumerate(e_ops):
                if isinstance(e_op, Qobj):
                    exp = expect(e_op, state)
                else:
                    exp = e_op(time, hamiltonian,
                               lambda *m_indices: self._ados(extended_state, *m_indices))
                result.expect[i].append(exp)

        options.normalize_output = False # important! since our "hamiltonian" is not hermitian
        sesolve(h_param, extended_rho0, tlist, e_ops=callback, options=options)
        return result

    def bath_heat_current(self, bath_index):
        r"""Bath heat current Q_B as defined in [Kato and Tanimura, J. Chem. Phys (2016)]:
            Q_B = -(d/dt) <H^k_B> ,
        where H^k_B is the Hamiltonian of the k-th reservoir (k=bath_index).

        Returns a callback function "callback(time, hamiltonian, ados)". Intended usage:
            heom = HEOM(...)
            heom.solve(rho0, tlist, e_ops=[heom.bath_heat_current(k), ...], ...)
        """

        # find which slice of the multi-index belongs to that bath
        j_min = sum(self._terms_per_bath[0:bath_index])
        j_max = sum(self._terms_per_bath[0:(bath_index+1)])
        # list of multi-indices with dimensions corresponding to all reservoirs
        multis = [[0] * terms for terms in self._terms_per_bath]

        coup_op = self.influence_phases[bath_index].coupling_op
        c_delta = self.influence_phases[bath_index].c_delta

        # e_ops-suitable callback function
        def callback(_, hamil, ados):
            result = 0
            for j in range(j_min, j_max, 1):
                multis[bath_index][j - j_min] = 1
                ado = ados(multis) # Note: ADO (Tanimura) = (-i)^level * ADO (our definition)
                multis[bath_index][j - j_min] = 0
                result -= 1j * self._freqs[j] * expect(coup_op, ado)
                # note: we leave out the following term because it always sums to zero, as
                #   long as the influence phase operator was constructed from a correlation
                #   function C with np.imag(C(0)) == 0.
                # result += 1j * (coup_op * vto(self._thetas[j] * otv(state))).tr()
                #   where state=ados(multis), vto=qutip.superoperators.vector_to_operator, otv=...

            if not c_delta:
                return result

            # TODO the following code tries to implement terms 3-5 of Eq. (28) in [Kato 2016]
            # (which are only relevant if c_delta is non-zero) but it has never been tried out...
            state = ados(multis)
            result -= c_delta * expect(commutator(commutator(hamil, coup_op), coup_op), state)

            for k_prime in range(len(self.influence_phases)):
                if k_prime == bath_index:
                    continue
                coup_op_prime = self.influence_phases[k_prime].coupling_op
                c_delta_prime = self.influence_phases[k_prime].c_delta
                b_kk = commutator(commutator(coup_op_prime, coup_op), coup_op)
                result += 1j * c_delta * c_delta_prime * expect(commutator(b_kk, coup_op_prime),
                                                                state)

                for j in range(len(multis[k_prime])):
                    multis[k_prime][j] = 1
                    ado = ados(multis)
                    multis[k_prime][j] = 0
                    result -= 1j * c_delta * expect(b_kk, ado)

            return result

        return callback

    def system_heat_current(self, bath_index):
        r"""System heat current Q_S as defined in [Kato and Tanimura, J. Chem. Phys (2016)]:
            Q_S = i * <[H^k_I, H_S(t)]> ,
        where H^k_I is the interaction Hamiltonian of the system with the k-th reservoir
        (k=bath_index) and H_S(t) is the system Hamiltonian.

        Returns a callback function "callback(time, hamiltonian, ados)". Intended usage:
            heom = HEOM(...)
            heom.solve(rho0, tlist, e_ops=[heom.system_heat_current(k), ...], ...)
        """

        multis = [[0] * terms for terms in self._terms_per_bath]
        coup_op = self.influence_phases[bath_index].coupling_op
        c_delta = self.influence_phases[bath_index].c_delta

        # e_ops-suitable callback function
        def callback(_, hamil, ados):
            a_k = commutator(hamil, coup_op)
            result = -c_delta * expect(commutator(a_k, coup_op), ados(multis)) # here, ados(multis)
            for j in range(len(multis[bath_index])):                           # = system state
                multis[bath_index][j] = 1
                ado = ados(multis)
                multis[bath_index][j] = 0
                result -= expect(a_k, ado)
            return result

        return callback

    def _ados(self, extended_state, m_indices):
        # Extracts the ADO indexed by the given multi-indices from the extended state
        # m_indices contains one multi-index per reservoir, we combine it to a single multi-index:
        multi = tuple([index for m_index in m_indices for index in m_index])
        return HEOM._extract_ado(extended_state, self._num_ados, self._multi_to_index[multi])

    @staticmethod
    def _extended_superoperator(superoperator, extended_dim, row, column):
        # Take the superoperator, which is a dim x dim matrix, and embed it in an
        # extended_dim x extended_dim block matrix, i.e., the extended matrix has dimensions
        # (dim*extended_dim) x (dim*extended_dim). The parameters row and column, with
        # 0 <= row, column < extended_dim, specify where in the extended matrix the block should
        # appear. All other entries of the extended matrix are zero.

        dim = superoperator.shape[0] # degrees of freedom in a single ADO
        if superoperator.shape[1] != dim:
            raise ValueError
        new_shape = (dim * extended_dim, dim * extended_dim)

        # (At least currently,) all QuTip data is stored as CSR sparse matrices.
        # We will rely on that in the following.
        if not isinstance(superoperator.data, csr_matrix):
            raise ValueError

        # shorthands
        indptr = superoperator.data.indptr
        indices = superoperator.data.indices
        data = superoperator.data.data

        # if indptr is e.g. [0,1,3] then we need to make [0,0,...,0,1,3,3,...,3]
        shifted_indptr = [0] * (row * dim) + list(indptr) +\
            [indptr[-1]] * ((extended_dim - row - 1) * dim)
        shifted_indices = indices + (column * dim)

        # Note that the extended matrix does not make sense any more as the matrix representation of
        # a superoperator, i.e., of an operator acting on an operator space (which must have shape
        # n**2 x n**2 for some integer n).
        # We will thus tell QuTiP that this is a normal operator.
        return Qobj(inpt=csr_matrix((data, shifted_indices, shifted_indptr), shape=new_shape),
                    shape=new_shape,
                    type="oper",
                    copy=True)

    @staticmethod
    def _diag_superoperator(superoperator, extended_dim):
        # Similar to _extended_superoperator, but puts superoperator on all diagonal blocks of the
        # extended block matrix

        dim = superoperator.shape[0] # degrees of freedom in a single ADO
        if superoperator.shape[1] != dim:
            raise ValueError
        new_shape = (dim * extended_dim, dim * extended_dim)
        if not isinstance(superoperator.data, csr_matrix):
            raise ValueError

        # shorthands
        indptr = superoperator.data.indptr
        indices = superoperator.data.indices
        data = superoperator.data.data

        # if indptr is e.g. [0,1,3] then we need to make [0,1,3,4,6,7,9,...]
        ran = range(extended_dim)
        new_indptr = np.concatenate([[0], *[indptr[1:] + k * indptr[-1] for k in ran]])
        new_indices = np.concatenate([indices + k * dim for k in ran])
        new_data = np.concatenate([data for _ in ran])

        return Qobj(inpt=csr_matrix((new_data, new_indices, new_indptr), shape=new_shape),
                    shape=new_shape,
                    type="oper",
                    copy=True)

    @staticmethod
    def _extended_operator(operator, extended_dim, row=0):
        # Analogous to _extended_superoperator, but for an operator (which is just a vector here)

        if operator.isoper:
            operator = operator_to_vector(operator)
        dim = operator.shape[0]
        new_shape = (dim * extended_dim, 1)

        if not isinstance(operator.data, csr_matrix):
            raise ValueError

        indptr = operator.data.indptr
        shifted_indptr = [0] * (row * dim) + list(indptr) +\
            [indptr[-1]] * ((extended_dim - row - 1) * dim)

        # In analogy to above, we claim that this is a normal ket, not an operator-ket.
        return Qobj(inpt=csr_matrix((operator.data.data, operator.data.indices, shifted_indptr),
                                    shape=new_shape),
                    shape=new_shape,
                    type="ket",
                    copy=True)

    @staticmethod
    def _extract_ado(extended_operator, extended_dim, row=0):
        # Inverse of _extended_operator

        if not extended_operator.isket:
            raise ValueError

        dim = int(extended_operator.shape[0] / extended_dim) # degrees of freedom in a single ADO
        hilbert_dim = int(np.sqrt(dim)) # dimension of the underlying Hilbert space
        new_shape = (dim, 1)

        if (dim * extended_dim != extended_operator.shape[0]) or (hilbert_dim**2 != dim):
            raise ValueError

        qobj = Qobj(inpt=extended_operator.data[(row * dim):((row+1) * dim), :],
                    dims=[[[hilbert_dim], [hilbert_dim]], [1]],
                    shape=new_shape,
                    type="operator-ket")
        return vector_to_operator(qobj)



class InfluencePhase:
    r"""The influence phase is a superoperator appearing in the Feynman-Vernon influence functional.
    The general form of the influence phase W(t) is
        W(t) = Qx * sum_J{ int_0^t d(tau) exp[nu_J(t-tau)] T_J(tau) } - c_delta * Qx^2 .
    Here, Q is the system coupling operator to the reservoir and Qx = [Q, ] the commutation
    superoperator. The frequencies nu_J are the characteristic frequencies of the free bath
    correlation function and T_J are superoperators as well (T_J(tau) is the superoperator in the
    interaction picture).

    This class serves as a container for all of these parameters, its member variables are:
        * coupling_op: the system coupling operator Q
        * coupling_superop: the commutation superoperator Qx
        * freqs: list of the frequencies nu_J
        * thetas: list of the corresponding superoperators T_J
        * num_terms: number of frequencies (=number of superoperators)
        * c_delta: the coefficient c_delta (which equals the coefficient c_delta
                                            appearing in the correlation function).

    Extracting these parameters from a given correlation function is slightly non-trivial.
    The function InfluencePhase.from_correlation_function serves that purpose."""

    def __init__(self, coupling_op, freqs, thetas, c_delta):
        r"""Initializes the member variables to the given values."""

        if len(freqs) != len(thetas):
            raise ValueError

        self.coupling_op = coupling_op
        self.coupling_superop = spre(coupling_op) - spost(coupling_op)
        self.freqs = freqs
        self.thetas = thetas
        self.num_terms = len(freqs)
        self.c_delta = c_delta

    @staticmethod
    def from_correlation_function(coupling_op, cfct):
        r"""Creates an influence phase from a given system coupling operator Q, and free bath
        correlation function cfct (of type CorrelationFunction)."""

        # The tricky bit is that 3 cases should be distinguished for each frequency nu_j appearing
        #   in the correlation function.
        # 1) nu_j is real. Then only one term is added to the influence phase, with superoperator
        #   -c_j spre(Q) + c_j* spost(Q). (* denotes complex conjugation.)
        # 2) nu_j is not real, and nu_j* does not appear in the correlation function. Then two terms
        #   are added, one with exponent nu_j and superoperator -c_j spre(Q), and one with exponent
        #   nu_j* and superoperator c_j* spost(Q).
        # 3) There is a j' so that nu_j' = nu_j*. Still only two terms are added in total, one with
        #   exponent nu_j and superoperator -c_j spre(Q) + (c_j')* spost(Q), and one with exponent
        #   nu_j* and superoperator c_j* spost(Q) - c_j' spre(Q).

        cfct_freq_and_coeff = list(zip(cfct.freqs, cfct.coeffs))
        result_freqs = []
        result_thetas = []

        spre_q = spre(coupling_op)
        spost_q = spost(coupling_op)

        # we use cfct_freq_and_coeff as a stack, removing all frequencies we have dealt with
        while cfct_freq_and_coeff: # empty list is falsy
            nu_j, c_j = cfct_freq_and_coeff.pop()

            # ----- Case 1 -----
            if np.isreal(nu_j):
                theta = -c_j * spre_q + np.conjugate(c_j) * spost_q
                result_freqs.append(nu_j)
                result_thetas.append(theta)
                continue

            j_prime = util.first_index(cfct_freq_and_coeff,
                                       lambda x: x[0] == np.conjugate(nu_j))
            # ----- Case 2 -----
            if j_prime is None:
                result_freqs.append(nu_j)
                result_thetas.append(-c_j * spre_q)
                result_freqs.append(np.conjugate(nu_j))
                result_thetas.append(np.conjugate(c_j) * spost_q)
            # ----- Case 3 -----
            else:
                nu_j_prime, c_j_prime = cfct_freq_and_coeff.pop(j_prime)
                theta_1 = -c_j * spre_q + np.conjugate(c_j_prime) * spost_q
                theta_2 = np.conjugate(c_j) * spost_q - c_j_prime * spre_q
                result_freqs.append(nu_j)
                result_thetas.append(theta_1)
                result_freqs.append(nu_j_prime)
                result_thetas.append(theta_2)

        return InfluencePhase(coupling_op, result_freqs, result_thetas, cfct.c_delta)


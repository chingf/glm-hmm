import numpy as np
import numpy.random as npr
from scipy.special import logsumexp

class SyntheticGLMHMM():
    def __init__(self, K, coefs, transitions):
        """
        A GLM-HMM with K states, where each state i is a different logistic
        regression model parameterized by coefs[i,:]. Transition probabilities
        are specified in transitions, a (K, K) matrix where value ij indicates
        the probability of transitioning from state i to state j. 
        """

        assert(K == coefs.shape[0])
        assert(K == transitions.shape[0] == transitions.shape[1])
        self.K = K
        self.coefs = coefs
        self.transitions = transitions
        Ps = 0.95*np.eye(K) + 0.05*npr.rand(K,K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

    def sample(self, inputs):
        """
        Inputs is a (T, M) matrix. Inputs should not include the previous
        choice as the synthetic model will populate that by itself. A column
        of 1 will also be added to each input to account for the bias term.
        """

        assert(self.coefs.shape[1] == (inputs.shape[1] + 2))
        states = []
        outputs = []
        prev_output = npr.choice(2)
        full_input = np.hstack((prev_output, inputs[0,:], 1))
        curr_state = npr.choice(self.K)
        curr_output = self._get_output(full_input, curr_state)
        for trial, input in enumerate(inputs):
            transitions = self._get_transition_matrix(full_input)
            full_input = np.hstack((prev_output, input, 1))
            prev_output = curr_output
            curr_state = npr.choice(
                self.K, p=transitions[curr_state,:]
                )
            curr_output = self._get_output(full_input, curr_state)
            states.append(curr_state)
            outputs.append(curr_output)
        states = np.array(states)
        outputs = np.array(outputs)
        return states, outputs

    def _get_output(self, input, state):
        logit_ps = input @ self.coefs[state,:]
        ps = 1/(1 + np.exp(-1*logit_ps))
        return npr.rand(1) < ps

    def _get_transition_matrix(self, input):
        return self.transitions

class SyntheticInputDrivenGLMHMM(SyntheticGLMHMM):
    def __init__(self, K, coefs, transition_coefs=None):
        """
        A GLM-HMM with K states, where each state i is a different logistic
        regression model parameterized by coefs[i,:]. Transition probabilities
        are specified through transition_coefs, a (K,M) matrix where row i
        provides the values for the softmax regression that determines the
        probability of transitioning to some state from state i. Bias terms
        are not used in transition coefficients and should not be included.
        For discrimination deltas, the weights in transition_coefs should be 0.
        """

        assert(K == coefs.shape[0])
        self.K = K
        self.coefs = coefs
        if transition_coefs is None:
            self.transition_coefs = npr.rand(K, coefs.shape[1])
        else:
            assert(transition_coefs.shape[0] == K)
            assert(transition_coefs.shape[1] == coefs.shape[1])
            self.transition_coefs = transition_coefs
        Ps = 0.95*np.eye(K) + 0.05*npr.rand(K,K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

    def _get_transition_matrix(self, input):
        log_Ps = self._get_log_transition_matrix(input)
        ps = np.exp(log_Ps)
        return ps 
    
    def _get_log_transition_matrix(self, input):
        log_Ps = self.log_Ps + (self.transition_coefs @ input)
        log_Ps = log_Ps - logsumexp(log_Ps, axis=1, keepdims=True)
        return log_Ps


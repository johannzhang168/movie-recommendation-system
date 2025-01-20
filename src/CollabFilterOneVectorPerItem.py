
import autograd.numpy as ag_np


from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets



class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''
    

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            mu=ag_np.ones(1),
            b_per_user=0.01 * random_state.randn(n_users), # FIX dimensionality
            c_per_item= 0.01 * random_state.randn(n_items), # FIX dimensionality
            U=0.001 * random_state.randn(n_users, self.n_factors), # FIX dimensionality
            V=0.001 * random_state.randn(n_items, self.n_factors), # FIX dimensionality
            )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''

        if mu is None:
            mu = self.param_dict['mu']
        if b_per_user is None:
            b_per_user = self.param_dict['b_per_user']
        if c_per_item is None:
            c_per_item = self.param_dict['c_per_item']
        if U is None:
            U = self.param_dict['U']
        if V is None:
            V = self.param_dict['V']

        N = user_id_N.size
        yhat_N = ag_np.ones(N)
        dot_products = ag_np.sum(U[user_id_N] * V[item_id_N], axis=1)

        yhat_N = mu + b_per_user[user_id_N] + c_per_item[item_id_N] + dot_products

        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''

        user_id_N = data_tuple[0]
        item_id_N = data_tuple[1]
        y_N = data_tuple[2]

        yhat_N = self.predict(user_id_N, item_id_N, **param_dict)
        mse = ag_np.sum((y_N - yhat_N) ** 2)
        # mse = ag_np.mean(sq_error)

        reg_term = self.alpha * (
            ag_np.sum((param_dict['U'][data_tuple[0]]) ** 2) +
            ag_np.sum((param_dict['V'][data_tuple[1]]) ** 2) +
            ag_np.sum((param_dict['b_per_user'][data_tuple[0]]) ** 2) +
            ag_np.sum((param_dict['c_per_item'][data_tuple[1]]) ** 2)
        )
        loss_total = mse + reg_term
        return loss_total  
    
      


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterOneVectorPerItem(
        n_epochs=5, batch_size=128, step_size=0.5,
        n_factors=50, alpha=0.0)
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)
    model.predict(test_tuple[0], test_tuple[1])

    # Final train error and validation error
    print("Train loss:", model.trace_mae_train[-1])
    print("Validation loss:", model.trace_mae_valid[-1])

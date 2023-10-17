from skorch import NeuralNet
from skorch.utils import to_device
from skorch.utils import to_tensor


# skorch objects
class DCSRO_Net(NeuralNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        return loss.mean()

    def score(self, X, y=None):
        y_pred = self.forward(to_tensor(X, self.device))
        # ipdb.set_trace()
        loss = super().get_loss(to_device(y_pred, self.device),
                                to_tensor(y, self.device),
                                X=to_tensor(X, self.device),
                                training=False)
        loss_value = loss.mean()
        return -loss_value



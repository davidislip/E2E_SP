import torch


def loss_fn(target, input, covariance_sqrt, risk_aversion, transaction_penalty):
    mu1_true, mu2_true = target
    y, w_1, y_2, w_2 = input
    """batched loss function for training the network"""
    # kl_loss = nn.KLDivLoss(reduction = 'batchmean')
    # loss = kl_loss(x_padm, x_exact)
    batch = w_1.size()[0]
    mu1_transpose = torch.permute(mu1_true[:,:,None], (0, 2, 1))
    mu2_transpose = torch.permute(mu2_true[:,:,None], (0, 2, 1))

    ret_1 = torch.squeeze(torch.bmm(mu1_transpose,  w_1[:,:,None]))
    ret_2 = torch.squeeze(torch.bmm(mu2_transpose,  w_2[:,:,None]))

    #take care of batching
    covariance_sqrt = covariance_sqrt[None, :,:]

    covariance_sqrt = covariance_sqrt.repeat(batch, 1, 1)
    sqrt_risk1 = torch.squeeze(torch.bmm(covariance_sqrt , w_1[:,:,None]))
    sqrt_risk2 = torch.squeeze(torch.bmm(covariance_sqrt ,w_2[:,:,None]))

    risk_1 = risk_aversion * torch.pow(torch.norm(sqrt_risk1, p=2, dim = -1), 2)
    risk_2 = risk_aversion * torch.pow(torch.norm(sqrt_risk2, p=2, dim = -1), 2)

    transaction_cost = transaction_penalty * (torch.norm(y, p = 1, dim = -1) + torch.norm(y_2, p = 1 , dim = -1))

    adjusted_returns = ret_1 + ret_2 - (risk_1 + risk_2 + transaction_cost)

    loss = -1*torch.mean(adjusted_returns)
    return loss

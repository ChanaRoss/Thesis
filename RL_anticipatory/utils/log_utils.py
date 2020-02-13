def log_values_step(cost, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts):
    avg_cost = cost.mean().item()
    avg_loss = reinforce_loss.item()
    avg_nll  = -log_likelihood.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}, loss: {}, nll: {}'.format(epoch, batch_id, avg_cost, avg_loss, avg_nll))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))

    # Log values to tensorboard
    if not opts['no_tensorboard']:
        tb_logger.add_scalar('loss/step/avg_cost_per_step', avg_cost, step)

        tb_logger.add_scalar('loss/step/actor_loss_per_step', avg_loss, step)
        tb_logger.add_scalar('loss/step/nll_per_step', avg_nll, step)

        tb_logger.add_scalar('optimizer/grad_norm', grad_norms[0], step)
        tb_logger.add_scalar('optimizer/grad_norm_clipped', grad_norms_clipped[0], step)

        if opts['baseline'] == 'critic':
            tb_logger.add_scalar('critic/critic_loss', bl_loss.item(), step)
            tb_logger.add_scalar('critic/critic_grad_norm', grad_norms[1], step)
            tb_logger.add_scalar('critic/critic_grad_norm_clipped', grad_norms_clipped[1], step)


def log_values_epoch(cost, epoch, log_likelihood, reinforce_loss, tb_logger, opts, model):
    avg_cost = cost
    avg_loss = reinforce_loss.item()
    avg_nll  = -log_likelihood

    # Log values to screen
    print('epoch: {} finished, avg_cost for epoch: {}, loss for epoch: {}, avg_nll for epoch: {}'.format(epoch, avg_cost, avg_loss, avg_nll))


    # Log values to tensorboard
    if not opts['no_tensorboard']:
        tb_logger.add_scalar('loss/epoch/avg_cost_per_epoch', avg_cost, epoch)

        tb_logger.add_scalar('loss/epoch/actor_loss_per_epoch', avg_loss, epoch)
        tb_logger.add_scalar('loss/epoch/nll_per_epoch', avg_nll, epoch)
        for name, param in model.named_parameters():
            tb_logger.add_histogram("parameters/{}".format(name), param.data, epoch)
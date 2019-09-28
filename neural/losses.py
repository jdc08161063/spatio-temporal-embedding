import torch


def discriminative_loss_static(y_batch, label_batch, config, device):
    """
    Parameters
    ----------
        y_batch: torch.tensor shape (batch_size, emb_dim, h, w)
        label_batch: torch.tensor<int> shape (batch_size, h, w)
        config: dict

    Returns
    -------
        total_loss: torch.tensor
            Discriminative loss for instance segmentation
    """
    delta_v = config['delta_v']
    delta_d = config['delta_d']
    batch_size = y_batch.size(0)

    total_v_loss = torch.tensor(0.0).to(device)
    total_d_loss = torch.tensor(0.0).to(device)
    total_reg_loss = torch.tensor(0.0).to(device)
    for i in range(batch_size):
        x = y_batch[i]
        label = label_batch[i]
        mask = label != 0
        emb_dim = x.size(0)

        # Mask out the background
        x = x[:, mask]
        label = label[mask]
        # Labels start at 1, make it start at 0
        label = (label - 1).byte()

        unique_labels = torch.unique(label)
        n_classes = len(unique_labels)
        assert torch.equal(unique_labels, torch.arange(n_classes, dtype=torch.uint8, device=device)), \
            'Unique labels are not consecutive'

        # Variance loss
        # Compute mean vector for each class
        indices = label.repeat(emb_dim, 1).long()
        mus = torch.zeros(emb_dim, n_classes, device=device).scatter_add(1, indices, x)
        counts = torch.zeros(emb_dim, n_classes, device=device).scatter_add(1, indices, torch.ones_like(x, device=device))
        mus = mus / counts  # shape (emb_dim, n_classes)

        v_loss = (torch.norm(x - torch.gather(mus, dim=1, index=indices), p=2, dim=0) - delta_v).clamp(min=0)
        v_loss = torch.pow(v_loss, 2)
        # Divide by pixel count of each instance
        v_loss /= torch.gather(counts[0, :], dim=0, index=label.long())
        v_loss = torch.sum(v_loss) / n_classes

        # Distance loss
        mus_repeat = mus.view(emb_dim, n_classes, 1).repeat(1, 1, n_classes)
        mus_repeat_t = mus.view(emb_dim, 1, n_classes).repeat(1, n_classes, 1)

        mus_matrix = (2 * delta_d - torch.norm(mus_repeat - mus_repeat_t, p=2, dim=0)).clamp(min=0)
        mus_matrix = torch.pow(mus_matrix, 2)

        # If contains more than one class
        if n_classes > 1:
            d_loss = mus_matrix[1 - torch.eye(n_classes, dtype=torch.uint8, device=device)].mean()
        else:
            d_loss = torch.tensor(0, dtype=torch.float, device=device)

        # Regularisation loss
        reg_loss = torch.norm(mus, p=2, dim=0).mean()

        total_v_loss += v_loss
        total_d_loss += d_loss
        total_reg_loss += reg_loss

    total_v_loss = config['lambda_v'] * total_v_loss / batch_size
    total_d_loss = config['lambda_d'] * total_d_loss / batch_size
    total_reg_loss = config['lambda_reg'] * total_reg_loss / batch_size

    losses = {'v_loss': total_v_loss,
              'd_loss': total_d_loss,
              'reg_loss': total_reg_loss}
    return losses


def discriminative_loss_static_loopy(y_batch, label_batch, config, device):
    """ Discriminative loss with loops, ignoring the background (id=0)

    Parameters
    ----------
        y_batch: torch.tensor shape (batch_size, emb_dim, h, w)
        label_batch: torch.tensor<int> shape (batch_size, h, w)
        config: dict with keys 'delta_v', 'delta_d', 'lambda_v', 'lambda_d', 'lambda_reg'

    Returns
    -------
        losses: dict
    """
    delta_v = config['delta_v']
    delta_d = config['delta_d']
    batch_size = y_batch.size(0)

    total_v_loss = torch.tensor(0.0).to(device)
    total_d_loss = torch.tensor(0.0).to(device)
    total_reg_loss = torch.tensor(0.0).to(device)

    for i in range(batch_size):
        # Variance loss
        x = y_batch[i]
        label = label_batch[i]

        v_loss = 0
        d_loss = 0
        reg_loss = 0

        unique_labels = torch.unique(label)
        # Remove background
        assert 0 in unique_labels
        unique_labels = unique_labels[1:]
        C = len(unique_labels)

        if C > 0:
            for c in unique_labels:
                x_masked = x[:, (label == c)]
                mu = x_masked.mean(dim=-1, keepdim=True)
                v_loss_current = (torch.norm(x_masked - mu, 2, dim=0) - delta_v).clamp(min=0)
                v_loss_current = torch.pow(v_loss_current, 2)
                v_loss += torch.mean(v_loss_current)

            v_loss /= C

            # Distance loss
            mus = []
            for c in unique_labels:
                x_masked = x[:, (label == c)]
                mu = x_masked.mean(dim=-1)
                mus.append(mu)

            # shape (C, emb_dim)
            mus = torch.stack(mus, dim=0)
            for i in range(C):
                for j in range(C):
                    if i == j:
                        continue
                    dist = (2 * delta_d - torch.norm(mus[i] - mus[j], 2)).clamp(min=0)
                    dist = torch.pow(dist, 2)
                    d_loss += dist

            d_loss /= torch.tensor(max(C * (C - 1), 1))  # so that d_loss is a torch.tensor (when C=1)

            # Regularisation loss
            for mu in mus:
                reg_loss += torch.norm(mu, 2)
            reg_loss /= C

        total_v_loss += v_loss
        total_d_loss += d_loss
        total_reg_loss += reg_loss

    total_v_loss = config['lambda_v'] * total_v_loss / batch_size
    total_d_loss = config['lambda_d'] * total_d_loss / batch_size
    total_reg_loss = config['lambda_reg'] * total_reg_loss / batch_size

    losses = {'v_loss': total_v_loss,
              'd_loss': total_d_loss,
              'reg_loss': total_reg_loss}
    return losses


def discriminative_loss_sequence_static(batch, output, config, device):
    """ Discriminative loss with loops, ignoring the background (id=0) (static frames)

    Parameters
    ----------
        batch: dict with key:
            instance_seg: torch.tensor<int> shape (batch_size, T, N_CLASSES, h, w)
        output: dict with key:
            y: torch.tensor shape (batch_size, T, emb_dim, h, w)
        config: dict with keys 'delta_v', 'delta_d', 'lambda_v', 'lambda_d', 'lambda_reg'

    Returns
    -------
        losses: dict
    """
    y_batch = output['y']
    label_batch = batch['instance_seg'].squeeze(2)
    seq_len = y_batch.size(1)

    losses = {'v_loss': torch.tensor(0.0).to(device),
              'd_loss': torch.tensor(0.0).to(device),
              'reg_loss': torch.tensor(0.0).to(device)}

    for t in range(seq_len):
        losses_t = discriminative_loss_static_loopy(y_batch[:, t], label_batch[:, t], config, device)
        for key in losses.keys():
            losses[key] += losses_t[key]

    for key in losses.keys():
        losses[key] /= seq_len

    return losses


def discriminative_loss_loopy(batch, output, config, device):
    """ Discriminative loss with loops, ignoring the background (id=0)

    Parameters
    ----------
        batch: dict with key:
            instance_seg: torch.tensor<int> shape (batch_size, T, N_CLASSES, h, w)
        output: dict with key:
            y: torch.tensor shape (batch_size, T, emb_dim, h, w)
        config: dict with keys 'delta_v', 'delta_d', 'lambda_v', 'lambda_d', 'lambda_reg', 'receptive_field'

    Returns
    -------
        losses: dict
    """
    y_batch = output['y']
    label_batch = batch['instance_seg'].squeeze(2)
    delta_v = config['delta_v']
    delta_d = config['delta_d']
    receptive_field = config['receptive_field']
    batch_size = y_batch.size(0)

    total_v_loss = torch.tensor(0.0).to(device)
    total_d_loss = torch.tensor(0.0).to(device)
    total_reg_loss = torch.tensor(0.0).to(device)

    for i in range(batch_size):
        # Variance loss
        x = y_batch[i].permute(1, 0, 2, 3)[:, (receptive_field-1):]
        label = label_batch[i][(receptive_field-1):]

        v_loss = 0
        d_loss = 0
        reg_loss = 0

        unique_labels = torch.unique(label)
        # Remove background
        assert 0 in unique_labels
        unique_labels = unique_labels[1:]
        C = len(unique_labels)

        if C > 0:
            for c in unique_labels:
                x_masked = x[:, (label == c)]
                mu = x_masked.mean(dim=-1, keepdim=True)
                v_loss_current = (torch.norm(x_masked - mu, 2, dim=0) - delta_v).clamp(min=0)
                v_loss_current = torch.pow(v_loss_current, 2)
                v_loss += torch.mean(v_loss_current)

            v_loss /= C

            # Distance loss
            mus = []
            for c in unique_labels:
                x_masked = x[:, (label == c)]
                mu = x_masked.mean(dim=-1)
                mus.append(mu)

            # shape (C, emb_dim)
            mus = torch.stack(mus, dim=0)
            for i in range(C):
                for j in range(C):
                    if i == j:
                        continue
                    dist = (2 * delta_d - torch.norm(mus[i] - mus[j], 2)).clamp(min=0)
                    dist = torch.pow(dist, 2)
                    d_loss += dist

            d_loss /= torch.tensor(max(C * (C - 1), 1))  # so that d_loss is a torch.tensor (when C=1)

            # Regularisation loss
            for mu in mus:
                reg_loss += torch.norm(mu, 2)
            reg_loss /= C

        total_v_loss += v_loss
        total_d_loss += d_loss
        total_reg_loss += reg_loss

    total_v_loss = config['lambda_v'] * total_v_loss / batch_size
    total_d_loss = config['lambda_d'] * total_d_loss / batch_size
    total_reg_loss = config['lambda_reg'] * total_reg_loss / batch_size

    losses = {'v_loss': total_v_loss,
              'd_loss': total_d_loss,
              'reg_loss': total_reg_loss}
    return losses


def mask_loss(batch, output):
    """ Cross-entropy loss

    Parameters
    ----------
        batch: dict with key:
            'instance_seg'
        output: dict with key:
            'mask'
    """
    b, t, c, h, w = output['mask_logits'].shape
    logits = output['mask_logits'].view(b*t, c, h, w)
    # TODO: N_CLASSES
    labels = (batch['instance_seg'].squeeze(2) > 0).view(b*t, h, w).long()
    mask_loss = torch.nn.functional.cross_entropy(input=logits, target=labels)

    losses = {'mask_loss': mask_loss}
    return losses


def motion_loss(batch, output, device):
    """
    Parameters
    ----------
        batch: dict with keys:
            position: torch.tensor (B, T, MAX_INSTANCES, 3)
            velocity: torch.tensor (B, T, MAX_INSTANCES, 3)
        output: dict with keys:
            position: torch.tensor (B, T, MAX_INSTANCES, 3)
            velocity: torch.tensor (B, T, MAX_INSTANCES, 3)
    """
    losses = {}
    position_loss = torch.tensor(0.0).to(device)
    velocity_loss = torch.tensor(0.0).to(device)

    batch_size = batch['img'].size(0)
    for i in range(batch_size):
        unique_ids = torch.unique(batch['instance_seg'][i])[1:].long()

        if len(unique_ids) > 0:
            # With the current model, we can only estimate z position (depth)
            position_loss += torch.dist(output['position'][i, :, unique_ids, 2],
                                        batch['position'][i, :, unique_ids, 2], p=2)
            # Only penalise 2D velocity
            velocity_loss += torch.dist(output['velocity'][i, :, unique_ids][:, :, [0, 2]],
                                        batch['velocity'][i, :, unique_ids][:, :, [0, 2]], p=2)

    position_loss /= batch_size
    velocity_loss /= batch_size

    losses['position_loss'] = position_loss
    losses['velocity_loss'] = velocity_loss
    return losses

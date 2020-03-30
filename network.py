import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, namedtuple

def make_conv_layer(in_channels,
                    out_channels,
                    filter_size,
                    non_linearity=True,
                    batch_norm=False,
                    atrou_rate=1):
    layers = []
    if filter_size == 1:
        padding_size = 0
    elif filter_size == 3:
        padding_size = atrou_rate
    else:
        raise

    if batch_norm:
        layers.append(
            ('conv', nn.Conv2d(in_channels, out_channels, filter_size,
                               padding=padding_size, dilation=atrou_rate, bias=False))
        )
        layers.append(('bn', nn.BatchNorm2d(out_channels, momentum=0.001, eps=0.001)))
    else:
        layers.append(
            ('conv', nn.Conv2d(in_channels, out_channels, filter_size,
                               padding=padding_size, dilation=atrou_rate, bias=True))
        )

    if non_linearity:
        layers.append(('elu', nn.ELU()))

    return nn.Sequential(OrderedDict(layers))

def make_conv_sep2d_layer(in_channels,
                          out_channels,
                          channel_multiplier,
                          filter_size,
                          filter_size_2=None,
                          batch_norm=False,
                          atrou_rate=1):
    """Use separable convolutions."""
    raise NotImplementedError

class ResidualBlock(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       layer_name,
                       filter_size,
                       batch_norm=False,
                       divide_channels_by=2,
                       atrou_rate=1,
                       channel_multiplier=0,
                       dropout_keep_prob=1.0):
        """A separable resnet block."""
        super().__init__()

        self.batch_norm = batch_norm
        self.dropout_keep_prob = dropout_keep_prob
        self.channel_multiplier = channel_multiplier

        if batch_norm:
            self.bn = nn.BatchNorm2d(in_channels, momentum=0.001, eps=0.001)
        self.elu = nn.ELU()

        # 1x1 with half size
        self.conv_1x1h = make_conv_layer(in_channels=in_channels,
                                         out_channels=in_channels // divide_channels_by,
                                         filter_size=1,
                                         non_linearity=True,
                                         batch_norm=batch_norm)

        # 3x3 with half size
        if channel_multiplier == 0:
            self.conv_3x3h = make_conv_layer(in_channels=in_channels // divide_channels_by,
                                             out_channels=in_channels // divide_channels_by,
                                             filter_size=filter_size,
                                             non_linearity=True,
                                             batch_norm=batch_norm,
                                             atrou_rate=atrou_rate)
        else:
            self.conv_sep3x3h = make_conv_sep2d_layer(in_channels=in_channels // divide_channels_by,
                                                      out_channels=in_channels // divide_channels_by,
                                                      channel_multiplier=channel_multiplier,
                                                      filter_size=filter_size,
                                                      batch_norm=batch_norm,
                                                      atrou_rate=atrou_rate)

        # 1x1 back to normal size without relu
        self.conv_1x1 = make_conv_layer(in_channels=in_channels // divide_channels_by,
                                        out_channels=out_channels,
                                        filter_size=1,
                                        non_linearity=False,
                                        batch_norm=False)

        if dropout_keep_prob < 1.0:
            self.dropout = nn.Dropout(1-dropout_keep_prob)

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(x)
            out = self.elu(out)
        else:
            out = self.elu(x)

        out = self.conv_1x1h(out)
        if self.channel_multiplier == 0:
            out = self.conv_3x3h(out)
        else:
            out = self.conv_sep3x3h(out)
        out = self.conv_1x1(out)

        if self.dropout_keep_prob < 1.0:
            out = self.dropout(out)

        out += x
        return out


def make_two_dim_resnet(num_features,
                        num_predictions=1,
                        num_channels=32,
                        num_layers=2,
                        filter_size=3,
                        final_non_linearity=False,
                        batch_norm=False,
                        atrou_rates=None,
                        channel_multiplier=0,
                        divide_channels_by=2,
                        resize_features_with_1x1=False,
                        dropout_keep_prob=1.0):
    if atrou_rates is None: atrou_rates = [1]

    layers = []
    non_linearity = True
    for i_layer in range(num_layers):
        in_channels = num_channels
        out_channels = num_channels
        curr_atrou_rate = atrou_rates[i_layer % len(atrou_rates)]

        if i_layer == 0:
            in_channels = num_features
        if i_layer == num_layers - 1:
            out_channels = num_predictions
            non_linearity = final_non_linearity

        if i_layer == 0 or i_layer == num_layers - 1:
            layer_name = f'conv{i_layer+1}'
            initial_filter_size = 1 if resize_features_with_1x1 else filter_size
            conv_layer = make_conv_layer(in_channels=in_channels,
                                        out_channels=out_channels,
                                        filter_size=initial_filter_size,
                                        non_linearity=non_linearity,
                                        atrou_rate=curr_atrou_rate)
        else:
            layer_name = f'res{i_layer+1}'
            conv_layer = ResidualBlock(in_channels=in_channels,
                                       out_channels=out_channels,
                                       layer_name=layer_name,
                                       filter_size=filter_size,
                                       batch_norm=batch_norm,
                                       atrou_rate=curr_atrou_rate,
                                       channel_multiplier=channel_multiplier,
                                       divide_channels_by=divide_channels_by,
                                       dropout_keep_prob=dropout_keep_prob)
        layers.append((layer_name, conv_layer))
    
    return nn.Sequential(OrderedDict(layers))

class ContactsNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        network_2d_deep = config.network_2d_deep
        output_dimension = config.num_bins
        if config.is_ca_feature:
            num_features = 12
        else:
            num_features = 1878
        threshold = 8
        self.quant_threshold = int((threshold - config.min_range) * config.num_bins / float(config.max_range))

        # total 220 residual blocks with dilated convolutions
        if network_2d_deep.extra_blocks:
            # 7 groups of 4 blocks with 256 channels, cycling through dilations 1,2,4,8.
            self.Deep2DExtra = make_two_dim_resnet(
                num_features=num_features,
                num_predictions=2 * network_2d_deep.num_filters,
                num_channels=2 * network_2d_deep.num_filters,
                num_layers=network_2d_deep.extra_blocks * network_2d_deep.num_layers_per_block,
                filter_size=3,
                batch_norm=network_2d_deep.use_batch_norm,
                final_non_linearity=True,
                atrou_rates=[1, 2, 4, 8],
                dropout_keep_prob=1.0
            )
            num_features = 2 * network_2d_deep.num_filters

        # 48 groups of 4 blocks with 128 channels, cycling through dilations 1,2,4,8.
        self.Deep2D = make_two_dim_resnet(
            num_features=num_features,
            num_predictions=network_2d_deep.num_filters if config.reshape_layer else output_dimension,
            num_channels=network_2d_deep.num_filters,
            num_layers=network_2d_deep.num_blocks * network_2d_deep.num_layers_per_block,
            filter_size=3,
            batch_norm=network_2d_deep.use_batch_norm,
            final_non_linearity=config.reshape_layer,
            atrou_rates=[1, 2, 4, 8],
            dropout_keep_prob=1.0
        )

        if config.reshape_layer:
            self.output_reshape_1x1h = make_conv_layer(
                in_channels=network_2d_deep.num_filters,
                out_channels=output_dimension,
                filter_size=1,
                non_linearity=False,
                batch_norm=network_2d_deep.use_batch_norm
            )

        if config.position_specific_bias_size:
            b = nn.Parameter(torch.zeros(config.position_specific_bias_size, output_dimension))
            self.register_parameter('position_specific_bias', b)

        embed_dim = 2*network_2d_deep.num_filters
        if config.collapsed_batch_norm:
            self.collapsed_batch_norm = nn.BatchNorm1d(embed_dim, momentum=0.001)

        if config.filters_1d:
            l = []
            for i, nfil in enumerate(config.filters_1d):
                if config.collapsed_batch_norm:
                    l.append(nn.Sequential(
                        nn.Linear(embed_dim, nfil, bias=False),
                        nn.BatchNorm1d(nfil, momentum=0.001)
                    ))
                else:
                    l.append(nn.Linear(embed_dim, nfil))
                embed_dim = nfil
            self.collapsed_embed = nn.ModuleList(l)

        if config.torsion_multiplier > 0:
            self.torsion_logits = nn.Linear(embed_dim, config.torsion_bins * config.torsion_bins)

        if config.secstruct_multiplier > 0:
            self.secstruct = nn.Linear(embed_dim, 8)

        if config.asa_multiplier > 0:
            self.ASALogits = nn.Linear(embed_dim, 1)


    def build_crops_biases(self, bias_size, raw_biases, crop_x, crop_y):
        max_off_diag = torch.max((crop_x[:, 1] - crop_y[:, 0]).abs(), (crop_y[:, 1] - crop_x[:, 0]).abs()).max()
        padded_bias_size = max(bias_size, max_off_diag)

        biases = torch.cat((raw_biases, raw_biases[-1:, :].repeat(padded_bias_size - bias_size, 1)), 0)
        biases = torch.cat((biases[1:, :].flip(0), biases), 0)

        start_diag = crop_x[:, 0:1] - crop_y[:, 0:1]
        crop_size_x = (crop_x[:, 1] - crop_x[:, 0]).max()
        crop_size_y = (crop_y[:, 1] - crop_y[:, 0]).max()

        increment = torch.unsqueeze(-torch.arange(0, crop_size_y), 0).to(crop_x.device)
        row_offsets = start_diag + increment
        row_offsets += padded_bias_size - 1

        cropped_biases = torch.cat(
            [torch.cat(
                [
                    biases[i:i+crop_size_x, :].unsqueeze(0) for i in offsets
                ], 0).unsqueeze(0)
                for offsets in row_offsets
            ], 0) # B*crop_y*crop_x*D
        cropped_biases = cropped_biases.permute(0, 3, 1, 2) # B*D*crop_y*crop_x

        return cropped_biases

    def forward(self, x, crop_x, crop_y):
        config = self.config

        out = self.Deep2DExtra(x)
        contact_pre_logits = self.Deep2D(out)

        if config.reshape_layer:
            contact_logits = self.output_reshape_1x1h(contact_pre_logits)
        else:
            contact_logits = contact_pre_logits

        if config.position_specific_bias_size:
            biases = self.build_crops_biases(config.position_specific_bias_size, self.position_specific_bias, crop_x, crop_y)
            contact_logits += biases # BxDxLxL

        contact_logits = contact_logits.permute(0, 2, 3, 1) # to NHWC shape
        distance_probs = nn.functional.softmax(contact_logits, -1) # BxLxLxD
        contact_probs = distance_probs[:, :, :, :self.quant_threshold].sum(-1) # BxLxL

        results = {
            'distance_probs': distance_probs,
            'contact_probs': contact_probs
        }

        if (config.secstruct_multiplier > 0 or
            config.asa_multiplier > 0 or
            config.torsion_multiplier > 0):
            collapse_dim = 2
            join_dim = 1

            embedding_1d = torch.cat((
                torch.cat((contact_pre_logits.max(collapse_dim)[0], contact_pre_logits.mean(collapse_dim)), join_dim),
                torch.cat((contact_pre_logits.max(collapse_dim+1)[0], contact_pre_logits.mean(collapse_dim+1)), join_dim)
            ), collapse_dim) # Bx2Dx2L

            if config.collapsed_batch_norm:
                embedding_1d = self.collapsed_batch_norm(embedding_1d)

            embedding_1d = embedding_1d.permute(0, 2, 1) # Bx2Lx2D
            for i, _ in enumerate(config.filters_1d):
                embedding_1d = self.collapsed_embed[i](embedding_1d)

            if config.torsion_multiplier > 0:
                torsion_logits = self.torsion_logits(embedding_1d)
                torsion_output = nn.functional.softmax(torsion_logits, -1)
                results['torsion_probs'] = torsion_output

            if config.secstruct_multiplier > 0:
                sec_logits = self.secstruct(embedding_1d)
                sec_output = nn.functional.softmax(sec_logits, -1)
                results['secstruct_probs'] = sec_output

            if config.asa_multiplier > 0:
                asa_logits = self.ASALogits(embedding_1d)
                asa_output = nn.functional.relu(asa_logits)
                results['asa_output'] = asa_output

        return results

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

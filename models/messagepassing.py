from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists
import torch
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct
from e3nn.nn import Gate
# from torch_scatter import scatter


def scatterd0(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # special case of torch_scatter.scatter with dim=0
    out = src.new_zeros(dim_size, src.shape[1])
    index = index.reshape(-1, 1).expand_as(src)
    return out.scatter_add_(0, index, src)


class Convolution(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        num_neighbors
    ) -> None:
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.num_neighbors = num_neighbors

        self.sc = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output)
        self.lin1 = FullyConnectedTensorProduct(self.irreps_node_input, self.irreps_node_attr, self.irreps_node_input)

        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(self.irreps_node_input):
            for j, (_, ir_edge) in enumerate(self.irreps_edge_attr):
                for ir_out in ir_in * ir_edge:
                    if ir_out in self.irreps_node_output or ir_out == o3.Irrep(0, 1):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        assert irreps_mid.dim > 0, f"irreps_node_input={self.irreps_node_input} time irreps_edge_attr={self.irreps_edge_attr} produces nothing in irreps_node_output={self.irreps_node_output}"

        instructions = [
            (i_1, i_2, p[i_out], mode, train)
            for i_1, i_2, i_out, mode, train in instructions
        ]

        tp = TensorProduct(
            self.irreps_node_input,
            self.irreps_edge_attr,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet(
            fc_neurons + [tp.weight_numel],
            torch.sin
        )
        self.tp = tp

        self.sc_edges = FullyConnectedTensorProduct(irreps_mid, self.irreps_edge_attr, self.irreps_edge_attr)

        self.lin2 = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_node_output)
        self.alpha = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, "0e")
        with torch.no_grad():
            self.alpha.weight.zero_()
        assert self.alpha.output_mask[0] == 1.0, f"irreps_mid={irreps_mid} and irreps_node_attr={self.irreps_node_attr} are not able to generate scalars"

    def forward(self, node_input, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        weight = self.fc(edge_scalars)

        node_self_connection = self.sc(node_input, node_attr)                                                   # -> irreps_node_output
        node_features = self.lin1(node_input, node_attr)                                                        # -> irreps_node_input

        edge_features = self.tp(node_features[edge_src], edge_attr, weight)                                     # -> irreps_mid
        node_features = scatterd0(edge_features, edge_dst, dim_size=node_input.shape[0]).div(self.num_neighbors**0.5)  # -> irreps_mid

        node_conv_out = self.lin2(node_features, node_attr)                                                     # -> irreps_node_output
        alpha = self.alpha(node_features, node_attr)                                                            # -> 0e
        m = self.sc.output_mask
        alpha = (1 - m) + alpha * m                                                                             # -> 0e

        node_conv_out = node_self_connection + alpha * node_conv_out                                            # -> irreps_node_output
        
        edge_conv_out = (edge_attr + self.sc_edges(edge_features, edge_attr).div(self.num_neighbors**0.5))      # -> irreps_edge_attr
        return node_conv_out, edge_conv_out                                                                     # -> irreps_node_output, irreps_edge_attr



class Compose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x, y = self.first(*input)
        return self.second(x), y


class MessagePassing(torch.nn.Module):
    def __init__(
        self,
        irreps_node_sequence,
        irreps_node_attr,
        irreps_edge_attr,
        fc_neurons,
        num_neighbors,
    ) -> None:
        super().__init__()
        self.num_neighbors = num_neighbors

        irreps_node_sequence = [o3.Irreps(irreps) for irreps in irreps_node_sequence]
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)

        act = {
            1: torch.sin,
            -1: torch.tanh,
        }
        act_gates = {
            1: torch.sin,
            -1: torch.tanh,
        }

        self.layers = torch.nn.ModuleList()

        self.irreps_node_sequence = [irreps_node_sequence[0]]
        irreps_node = irreps_node_sequence[0]

        for irreps_node_hidden in irreps_node_sequence[1:-1]:
            irreps_scalars = o3.Irreps([
                (mul, ir)
                for mul, ir in irreps_node_hidden
                if ir.l == 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
            ]).simplify()
            irreps_gated = o3.Irreps([
                (mul, ir)
                for mul, ir in irreps_node_hidden
                if ir.l > 0 and tp_path_exists(irreps_node, self.irreps_edge_attr, ir)
            ])
            if irreps_gated.dim > 0:
                if tp_path_exists(irreps_node, self.irreps_edge_attr, "0e"):
                    ir = "0e"
                elif tp_path_exists(irreps_node, self.irreps_edge_attr, "0o"):
                    ir = "0o"
                else:
                    raise ValueError(f"irreps_node={irreps_node} times irreps_edge_attr={self.irreps_edge_attr} is unable to produce gates needed for irreps_gated={irreps_gated}")
            else:
                ir = None
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

            gate = Gate(
                irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],    # scalar
                irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated                                                # gated tensors
            )
            conv = Convolution(
                irreps_node,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                fc_neurons,
                num_neighbors
            )
            self.layers.append(Compose(conv, gate))
            irreps_node = gate.irreps_out
            self.irreps_node_sequence.append(irreps_node)

        irreps_node_output = irreps_node_sequence[-1]
        self.layers.append(
            Convolution(
                irreps_node,
                
                self.irreps_node_attr,
                self.irreps_edge_attr,
                irreps_node_output,
                fc_neurons,
                num_neighbors
            )
        )
        self.irreps_node_sequence.append(irreps_node_output)

        self.irreps_node_input = self.irreps_node_sequence[0]
        self.irreps_node_output = self.irreps_node_sequence[-1]

    def forward(self, node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        for lay in self.layers:
            node_features, edge_attr = lay(node_features, node_attr, edge_src, edge_dst, edge_attr, edge_scalars)

        return node_features, edge_attr
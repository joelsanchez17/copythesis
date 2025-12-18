import torch
from torch import nn
import itertools

class MonomialMatrixModule(nn.Module):
    def __init__(self, shape, num_vars, monomial_order, symmetric=False):
        super().__init__()
        self.num_vars = num_vars
        self.monomial_order = monomial_order
        self.symmetric = symmetric
        self.shape = shape
        # coeffs_shape = (shape[0] * (shape[0] + 1) // 2, shape[1]) if symmetric else shape
        # self.coeffs = nn.Parameter(torch.randn(*shape, self.calculate_monomials()))
        self.coeffs = nn.Parameter(torch.randn(*shape, self.calculate_monomials()))
        if symmetric and shape[0] != shape[1]:
            raise ValueError("Symmetric matrices must be square")
        if symmetric:
            self.coeffs = nn.Parameter(self.coeffs.transpose(2,0).tril().transpose(0,2))
    def calculate_monomials(self):
        # Calculate the total number of monomials
        total_monomials = 1
        for i in range(1, self.monomial_order + 1):
            total_monomials += len(list(itertools.combinations_with_replacement(range(self.num_vars), i)))
        return total_monomials

    def forward(self, x):
        x = x.unsqueeze(-1)  # make it a column vector

        # Create a list to hold the monomials
        monomials = [torch.ones_like(x[0]).squeeze()] # for the constant term

        for i in range(1, self.monomial_order+1):
            for comb in itertools.combinations_with_replacement(range(self.num_vars), i):
                monomial = torch.prod(x[list(comb)])
                monomials.append(monomial)

        v = torch.stack(monomials)

        matrix = torch.einsum('jki,i->jk', self.coeffs, v)
        
        if self.symmetric:
            matrix = matrix + matrix.triu(1).t()

        return matrix
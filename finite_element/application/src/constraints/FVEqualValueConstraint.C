#include "FVEqualValueConstraint.h"

#include "Assembly.h"
#include "MooseVariableField.h"
#include "MooseVariableFE.h"
#include "SystemBase.h"

registerMooseObject("cavityApp", FVEqualValueConstraint);

InputParameters
FVEqualValueConstraint::validParams()
{
  InputParameters params = OldEqualValueConstraint::validParams();
  params.addClassDescription(
      "EqualValueConstraint (mortar, non-AD) with finite-volume-compatible "
      "Jacobian assembly; use for periodic BCs on FV/INSFV variables.");
  return params;
}

FVEqualValueConstraint::FVEqualValueConstraint(const InputParameters & parameters)
  : OldEqualValueConstraint(parameters)
{
}

void
FVEqualValueConstraint::computeJacobian(Moose::MortarType mortar_type)
{
  // Identical to MortarConstraint::computeJacobian except the phi lookup
  // casts to MooseVariableField<Real> (valid for FE *and* FV variables)
  // instead of MooseVariableFE<Real> (UB/segfault for FV variables).
  std::size_t test_space_size = 0;
  typedef Moose::ConstraintJacobianType JType;
  typedef Moose::MortarType MType;
  std::array<JType, 3> jacobian_types;

  switch (mortar_type)
  {
    case MType::Secondary:
      test_space_size = _secondary_var.dofIndices().size();
      jacobian_types = {
          {JType::SecondarySecondary, JType::SecondaryPrimary, JType::SecondaryLower}};
      break;

    case MType::Primary:
      test_space_size = _primary_var.dofIndicesNeighbor().size();
      jacobian_types = {{JType::PrimarySecondary, JType::PrimaryPrimary, JType::PrimaryLower}};
      break;

    case MType::Lower:
      test_space_size = _var ? _var->dofIndicesLower().size() : 0;
      jacobian_types = {{JType::LowerSecondary, JType::LowerPrimary, JType::LowerLower}};
      break;
  }

  auto & ce = _assembly.couplingEntries();
  for (const auto & it : ce)
  {
    MooseVariableFEBase & ivariable = *(it.first);
    MooseVariableFEBase & jvariable = *(it.second);

    unsigned int ivar = ivariable.number();
    unsigned int jvar = jvariable.number();

    switch (mortar_type)
    {
      case MType::Secondary:
        if (ivar != _secondary_var.number())
          continue;
        break;

      case MType::Primary:
        if (ivar != _primary_var.number())
          continue;
        break;

      case MType::Lower:
        if (!_var || _var->number() != ivar)
          continue;
        break;
    }

    std::array<size_t, 3> shape_space_sizes{{jvariable.dofIndices().size(),
                                             jvariable.dofIndicesNeighbor().size(),
                                             jvariable.dofIndicesLower().size()}};
    std::array<const VariablePhiValue *, 3> phis;
    std::array<const VariablePhiGradient *, 3> grad_phis;
    if (jvariable.isVector())
      mooseError("FVEqualValueConstraint does not support vector variables");
    else if (jvariable.isFV())
    {
      // THE FIX: an FV primal variable has NO lower-dimensional dofs, so the
      // "Lower" phi (index 2) is never dereferenced (dofIndicesLower()==0 ->
      // empty _j loop).  Calling phiLower()/gradPhi*() on an FV variable
      // throws ("Lower dimensional element support not implemented"), so we
      // must NOT build them.  Only phiFace/phiFaceNeighbor are needed and are
      // implemented by MooseVariableFV (constant-1 cell basis).
      const auto & temp_var = static_cast<MooseVariableField<Real> &>(jvariable);
      phis = {{&temp_var.phiFace(), &temp_var.phiFaceNeighbor(), nullptr}};
      grad_phis = {{nullptr, nullptr, nullptr}};
    }
    else
    {
      const auto & fe_var = static_cast<MooseVariableFE<Real> &>(jvariable);
      phis = {{&fe_var.phiFace(), &fe_var.phiFaceNeighbor(), &fe_var.phiLower()}};
      grad_phis = {
          {&fe_var.gradPhiFace(), &fe_var.gradPhiFaceNeighbor(), &fe_var.gradPhiLower()}};
    }

    for (MooseIndex(3) type_index = 0; type_index < 3; ++type_index)
    {
      const auto jacobian_type = jacobian_types[type_index];

      prepareMatrixTagLower(_assembly, ivar, jvar, jacobian_type);

      _phi = phis[type_index];
      _grad_phi = grad_phis[type_index];

      for (_i = 0; _i < test_space_size; _i++)
        for (_j = 0; _j < shape_space_sizes[type_index]; _j++)
          for (_qp = 0; _qp < _qrule_msm->n_points(); _qp++)
            _local_ke(_i, _j) +=
                _JxW_msm[_qp] * _coord[_qp] * computeQpJacobian(jacobian_type, jvar);
      accumulateTaggedLocalMatrix();
    }
  }
}

//* App-local fix: mortar EqualValue constraint whose Jacobian assembly is
//* FINITE-VOLUME compatible.
//*
//* Why: framework MortarConstraint::computeJacobian() does
//*     static_cast<MooseVariableFE<Real> &>(jvariable).phiFace()
//* which is invalid (segfault) when the primal variable is an FV variable
//* (e.g. INSFVVelocityVariable).  But phiFace()/phiFaceNeighbor()/phiLower()
//* are pure-virtual on the common base MooseVariableField<Real> and ARE
//* implemented by MooseVariableFV (constant-1 basis, one dof per cell), so
//* the correct cast makes mortar periodic BCs work for FV variables.
//*
//* Residual/Jacobian formulas = OldEqualValueConstraint (non-AD; the AD
//* mortar path crashes evaluating INSFV variables).

#pragma once

#include "OldEqualValueConstraint.h"

class FVEqualValueConstraint : public OldEqualValueConstraint
{
public:
  static InputParameters validParams();

  FVEqualValueConstraint(const InputParameters & parameters);

protected:
  // re-implementation of MortarConstraint::computeJacobian with the
  // FV-compatible variable cast
  void computeJacobian(Moose::MortarType mortar_type) override;
};

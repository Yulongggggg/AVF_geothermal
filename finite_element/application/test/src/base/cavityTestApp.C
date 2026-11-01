//* This file is part of the MOOSE framework
//* https://mooseframework.inl.gov
//*
//* All rights reserved, see COPYRIGHT for full restrictions
//* https://github.com/idaholab/moose/blob/master/COPYRIGHT
//*
//* Licensed under LGPL 2.1, please see LICENSE for details
//* https://www.gnu.org/licenses/lgpl-2.1.html
#include "cavityTestApp.h"
#include "cavityApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "MooseSyntax.h"

InputParameters
cavityTestApp::validParams()
{
  InputParameters params = cavityApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

cavityTestApp::cavityTestApp(const InputParameters & parameters) : MooseApp(parameters)
{
  cavityTestApp::registerAll(
      _factory, _action_factory, _syntax, getParam<bool>("allow_test_objects"));
}

cavityTestApp::~cavityTestApp() {}

void
cavityTestApp::registerAll(Factory & f, ActionFactory & af, Syntax & s, bool use_test_objs)
{
  cavityApp::registerAll(f, af, s);
  if (use_test_objs)
  {
    Registry::registerObjectsTo(f, {"cavityTestApp"});
    Registry::registerActionsTo(af, {"cavityTestApp"});
  }
}

void
cavityTestApp::registerApps()
{
  registerApp(cavityApp);
  registerApp(cavityTestApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
// External entry point for dynamic application loading
extern "C" void
cavityTestApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  cavityTestApp::registerAll(f, af, s);
}
extern "C" void
cavityTestApp__registerApps()
{
  cavityTestApp::registerApps();
}

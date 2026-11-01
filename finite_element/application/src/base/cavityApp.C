#include "cavityApp.h"
#include "Moose.h"
#include "AppFactory.h"
#include "ModulesApp.h"
#include "MooseSyntax.h"

InputParameters
cavityApp::validParams()
{
  InputParameters params = MooseApp::validParams();
  params.set<bool>("use_legacy_material_output") = false;
  params.set<bool>("use_legacy_initial_residual_evaluation_behavior") = false;
  return params;
}

cavityApp::cavityApp(const InputParameters & parameters) : MooseApp(parameters)
{
  cavityApp::registerAll(_factory, _action_factory, _syntax);
}

cavityApp::~cavityApp() {}

void
cavityApp::registerAll(Factory & f, ActionFactory & af, Syntax & syntax)
{
  ModulesApp::registerAllObjects<cavityApp>(f, af, syntax);
  Registry::registerObjectsTo(f, {"cavityApp"});
  Registry::registerActionsTo(af, {"cavityApp"});

  /* register custom execute flags, action syntax, etc. here */
}

void
cavityApp::registerApps()
{
  registerApp(cavityApp);
}

/***************************************************************************************************
 *********************** Dynamic Library Entry Points - DO NOT MODIFY ******************************
 **************************************************************************************************/
extern "C" void
cavityApp__registerAll(Factory & f, ActionFactory & af, Syntax & s)
{
  cavityApp::registerAll(f, af, s);
}
extern "C" void
cavityApp__registerApps()
{
  cavityApp::registerApps();
}

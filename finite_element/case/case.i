mesh_file = mesh.msh
perm_block_file = 'permeability.i'
prod_bh_file = producer.bh
prod_pt = '44 213.6 0'
inj_T = 293.2
prod_p = 2.22556e+07

prod_line_length = 1.0
injection_mass_rate = 1.1

xmin = -400
xmax =  400
ymin = -500
ymax =  500
ml_pt = '${xmin} 0 0'
mr_pt = '${xmax} 0 0'
mb_pt = '0 ${ymin} 0'
mt_pt = '0 ${ymax} 0'

inj_pt  = '0 -250 0'
Lz = 5

pp_ic = 30e6
T_ic  = 354.15
y_ref  = 0.0
pp_ref = ${pp_ic}
T_ref  = ${T_ic}
epv_ref = 0

sigmaV_insitu_ref = 78.6e6

sigmaH_insitu_ref = 103.4e6

sigma3_insitu_ref = 70.7e6

dpdy = -9810

dTdy = -0.023

dsigmaV_insitu_dy = -2.72e4

dsigmaH_insitu_dy = -4.88e4

dsigma3_insitu_dy = -2.65e4

phi_block = 3.9e-2

phi_cap_bed = 1e-3
k_cap_bed = 1e-18
kth_cap_bed = 2.8
rho_s_block = 2800
cp_s_block  = 1000
kth_block   = 2.8
epv_coeff = 0.541
T_coeff = -1.623e-05
P_coeff = 7.9160516129e-12
epv_coeff_cap_bed = 0.579
T_coeff_cap_bed = -1.737e-05
P_coeff_cap_bed = 8.47207741935e-12
rho_s_cap = ${rho_s_block}
cp_s_cap  = ${cp_s_block}

E_block  = 31e9
nu_block = 0.32
alphaT_block = 1.0e-5
E_cap  = ${E_block}
nu_cap = ${nu_block}
biot = 0.58

nodeset_tolerance = 1e-3
prod_bhp_ramp_t_end = 2000

MW = 1e6
GJ = 1e9

kc_m_block = 2
kc_n_block = 3

peaceman_character = 1

dt0 = 1000
dt_growth = 2
dt_cutback = 0.7
dt_opt_its = 12
end_time = 3.16e8

nl_abs_tol = 1e-5
nl_rel_tol = 1e-5
l_tol = 1e-6
l_max_its = 1000
nl_max_its = 50

[Mesh]
  final_generator = mt_nodeset

  [from_file]
    type = FileMeshGenerator
    file = ${mesh_file}
  []
  [inj_nodeset]
    type = ExtraNodesetGenerator
    input = from_file
    new_boundary = injection_node
    coord = ${inj_pt}
    use_closest_node = true
    tolerance = ${nodeset_tolerance}
  []
  [ml_nodeset]
    type = ExtraNodesetGenerator
    input = inj_nodeset
    new_boundary = ml
    coord = ${ml_pt}
    use_closest_node = true
    tolerance = ${nodeset_tolerance}
  []
  [mr_nodeset]
    type = ExtraNodesetGenerator
    input = ml_nodeset
    new_boundary = mr
    coord = ${mr_pt}
    use_closest_node = true
    tolerance = ${nodeset_tolerance}
  []
  [mb_nodeset]
    type = ExtraNodesetGenerator
    input = mr_nodeset
    new_boundary = mb
    coord = ${mb_pt}
    use_closest_node = true
    tolerance = ${nodeset_tolerance}
  []
  [mt_nodeset]
    type = ExtraNodesetGenerator
    input = mb_nodeset
    new_boundary = mt
    coord = ${mt_pt}
    use_closest_node = true
    tolerance = ${nodeset_tolerance}
  []
[]

[GlobalParams]
  PorousFlowDictator = dictator
  displacements = 'disp_x disp_y'
[]

[Variables]
  [pp]
  []
  [T]
  []
  [disp_x]
    family = LAGRANGE
    order  = FIRST
  []
  [disp_y]
    family = LAGRANGE
    order  = FIRST
  []
[]

[ICs]
  [pp_ic_gradient]
    type = FunctionIC
    variable = pp
    function = pp_init
  []
  [T_ic_gradient]
    type = FunctionIC
    variable = T
    function = T_init
  []
  [T_stress_free_gradient]
    type = FunctionIC
    variable = T_stress_free
    function = T_init
  []
  [T_injection_ic]
    type = ConstantIC
    variable = T_injection
    value = ${inj_T}
  []
[]

[AuxVariables]
  [density]
    family = MONOMIAL
    order  = CONSTANT
  []
  [stress_xx]
    family=MONOMIAL
    order= CONSTANT
  []
  [stress_yy]
    family=MONOMIAL
    order= CONSTANT
  []
  [stress_zz]
    family=MONOMIAL
    order= CONSTANT
  []
  [porosity]
    family = MONOMIAL
    order  = CONSTANT
  []
  [permeability]
    family = MONOMIAL
    order  = CONSTANT
  []
  [perm_block_field]
    family = MONOMIAL
    order  = CONSTANT
  []
  [perm_block_A]
    family = MONOMIAL
    order  = CONSTANT
  []
  [insitu_pp]
  []
  [T_stress_free]
  []
  [T_injection]
  []
  [h_injection]
  []
[]

[AuxKernels]
  [density]
    type = PorousFlowPropertyAux
    variable = density
    property = density
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [sx]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_xx
    index_i = 0
    index_j = 0
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [sy]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_yy
    index_i = 1
    index_j = 1
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [sz]
    type = RankTwoAux
    rank_two_tensor = stress
    variable = stress_zz
    index_i = 2
    index_j = 2
    execute_on = 'INITIAL TIMESTEP_END'
  []
  [porosity]
    type = PorousFlowPropertyAux
    variable = porosity
    property = porosity
    execute_on = TIMESTEP_END
  []
  [permeability]
    type = PorousFlowPropertyAux
    variable = permeability
    property = permeability
    execute_on = TIMESTEP_END
  []
  [perm_block_from_file]
    type = FunctionAux
    variable = perm_block_field
    function = perm_block_fn
    block = 'block'
    execute_on = INITIAL
  []
  [perm_block_A_from_k]
    type = ParsedAux
    variable = perm_block_A
    coupled_variables = perm_block_field
    expression = 'perm_block_field * (1 - ${phi_block})^${kc_m_block} / (${phi_block}^${kc_n_block})'
    block = 'block'
    execute_on = INITIAL
  []
  [insitu_pp]
    type = FunctionAux
    execute_on = INITIAL
    variable = insitu_pp
    function = pp_init
  []
  [h_injection]
    type = SpecificEnthalpyAux
    variable = h_injection
    p = pp
    T = T_injection
    fp = Water97FluidProperties
    execute_on = 'INITIAL TIMESTEP_END'
  []
[]

[Functions]
  [perm_block_fn]
    type = PiecewiseMultilinear
    data_file = ${perm_block_file}
  []
  [prod_bhp]
    type = ParsedFunction
    expression = 'if(t < ${prod_bhp_ramp_t_end}, (${pp_ref}+${dpdy}*(y-${y_ref}))+(${prod_p}-(${pp_ref}+${dpdy}*(y-${y_ref})))*t/${prod_bhp_ramp_t_end}, ${prod_p})'
  []
  [pp_init]
    type = ParsedFunction
    expression = '${pp_ref} + ${dpdy}*(y - ${y_ref})'
  []
  [T_init]
    type = ParsedFunction
    expression = '${T_ref} + ${dTdy}*(y - ${y_ref})'
  []
  [injection_rate]
    type = ParsedFunction
    expression = '${injection_mass_rate}/${Lz}'
  []
  [sigmaV_insitu_init]
    type = ParsedFunction
    expression = '-(${sigmaV_insitu_ref} + ${dsigmaV_insitu_dy}*(y - ${y_ref}))'
  []
  [sigmaH_insitu_init]
    type = ParsedFunction
    expression = '-(${sigmaH_insitu_ref} + ${dsigmaH_insitu_dy}*(y - ${y_ref}))'
  []
  [sigmaV_eff]
    type = ParsedFunction
    expression = '-(${sigmaV_insitu_ref} + ${dsigmaV_insitu_dy}*(y - ${y_ref})) + ${biot}*(${pp_ref} + ${dpdy}*(y - ${y_ref}))'
  []
  [sigmaH_eff]
    type = ParsedFunction
    expression = '-(${sigmaH_insitu_ref} + ${dsigmaH_insitu_dy}*(y - ${y_ref})) + ${biot}*(${pp_ref} + ${dpdy}*(y - ${y_ref}))'
  []
  [pressV_mag]
    type = ParsedFunction
    expression = 'abs(-(${sigmaV_insitu_ref} + ${dsigmaV_insitu_dy}*(y - ${y_ref})))'
  []
  [pressH_mag]
    type = ParsedFunction
    expression = 'abs(-(${sigmaH_insitu_ref} + ${dsigmaH_insitu_dy}*(y - ${y_ref})))'
  []
  [sigma3_insitu_init]
    type = ParsedFunction
    expression = '-(${sigma3_insitu_ref} + ${dsigma3_insitu_dy}*(y - ${y_ref}))'
  []
  [sigma3_eff]
    type = ParsedFunction
    expression = '-(${sigma3_insitu_ref} + ${dsigma3_insitu_dy}*(y - ${y_ref})) + ${biot}*(${pp_ref} + ${dpdy}*(y - ${y_ref}))'
  []
[]

[PorousFlowFullySaturated]
  coupling_type = ThermoHydroMechanical
  gravity = '0 -9.81 0'
  biot_coefficient = ${biot}

  add_stress_aux = false
  add_darcy_aux = false
  porepressure = pp
  temperature = T
  displacements = 'disp_x disp_y'
  fp = Water97FluidProperties
  block = 'cap bed block'
  stabilization = full
  eigenstrain_names = 'thermal_contribution'
  fluid_properties_type = 'PorousFlowSingleComponentFluid'
  multiply_by_density = true
[]

[BCs]
  [far_r]
    type = Pressure
    boundary = r
    variable = disp_x
    function = pressH_mag
  []
  [far_t]
    type = Pressure
    boundary = t
    variable = disp_y
    function = pressV_mag
  []
  [far_l]
    type = Pressure
    boundary = l
    variable = disp_x
    function = pressH_mag
  []
  [pin_bottom_x]
    type = DirichletBC
    boundary = mb
    variable = disp_x
    value = 0
  []
  [roller_bottom_y]
    type = DirichletBC
    boundary = b
    variable = disp_y
    value = 0
  []
[]

[UserObjects]
  [produced_mass]
    type = PorousFlowSumQuantity
  []
  [produced_heat]
    type = PorousFlowSumQuantity
  []
[]

[DiracKernels]
  [inject_fluid]
    type = PorousFlowPointSourceFromPostprocessor
    mass_flux = inj_mass_flux_pp
    point = ${inj_pt}
    variable = pp
  []
  [source_h]
    type = PorousFlowPointEnthalpySourceFromPostprocessor
    variable = T
    mass_flux = inj_mass_flux_pp
    point = ${inj_pt}
    T_in = ${inj_T}
    pressure = pp
    fp = Water97FluidProperties
  []
  [fluid_production]
    type = PorousFlowPeacemanBorehole
    variable = pp
    SumQuantityUO = produced_mass
    point_file = ${prod_bh_file}
    function_of = pressure
    fluid_phase = 0
    bottom_p_or_t = prod_bhp
    unit_weight = '0 0 0'
    use_mobility = true
    line_length = ${prod_line_length}
    character = ${peaceman_character}
  []
  [remove_heat_at_production_well]
    type = PorousFlowPeacemanBorehole
    variable = T
    SumQuantityUO = produced_heat
    point_file = ${prod_bh_file}
    function_of = pressure
    fluid_phase = 0
    bottom_p_or_t = prod_bhp
    unit_weight = '0 0 0'
    use_mobility = true
    use_enthalpy = true
    line_length = ${prod_line_length}
    character = ${peaceman_character}
  []
[]

[Postprocessors]
  [prod_mass_kg_this_step]
    type = PorousFlowPlotQuantity
    uo = produced_mass
  []
  [heat_J_this_step]
    type = PorousFlowPlotQuantity
    uo = produced_heat
  []
  [dt_pp]
    type = TimestepSize
  []
  [m_prod_kgs]
    type = ParsedPostprocessor
    pp_names   = 'prod_mass_kg_this_step dt_pp'
    pp_symbols = 'm dt'
    expression = '(m/dt)*${Lz}'
  []
  [inj_mass_flux_pp]
    type = FunctionValuePostprocessor
    function = injection_rate
    execute_on = timestep_begin
  []
  [m_inj_kgs]
    type = ParsedPostprocessor
    pp_names   = 'inj_mass_flux_pp'
    pp_symbols = 'm'
    expression = 'm*${Lz}'
  []
  [Qprod_W]
    type = ParsedPostprocessor
    pp_names   = 'heat_J_this_step dt_pp'
    pp_symbols = 'Q dt'
    expression = '(Q/dt)*${Lz}'
  []
  [h_inj]
    type = PointValue
    variable = h_injection
    point = ${inj_pt}
  []
  [Qth_W]
    type = ParsedPostprocessor
    pp_names   = 'Qprod_W m_prod_kgs h_inj'
    pp_symbols = 'Qprod mprod hinj'

    expression = 'Qprod-mprod*hinj'
  []
  [Qth_MW]
    type = ParsedPostprocessor
    pp_names   = 'Qth_W'
    pp_symbols = 'P'
    expression = 'P/${MW}'
  []
  [net_heat_J_this_step]
    type = ParsedPostprocessor
    pp_names   = 'heat_J_this_step prod_mass_kg_this_step h_inj'
    pp_symbols = 'Eprod mprod hinj'
    expression = '(Eprod-mprod*hinj)*${Lz}'
  []
  [cum_heat_J]
    type = CumulativeValuePostprocessor
    postprocessor = net_heat_J_this_step
  []
  [cum_heat_GJ]
    type = ParsedPostprocessor
    pp_names   = 'cum_heat_J'
    pp_symbols = 'E'
    expression = 'E/${GJ}'
  []
  [T_prod]
    type = PointValue
    variable = T
    point = ${prod_pt}
  []
  [Qnet_W]
    type = ParsedPostprocessor
    pp_names   = 'Qth_W'
    pp_symbols = 'Q'
    expression = 'Q'
  []
  [Qnet_MW]
    type = ParsedPostprocessor
    pp_names   = 'Qnet_W'
    pp_symbols = 'P'
    expression = 'P/${MW}'
  []
  [pp_prod]
    type = PointValue
    variable = pp
    point = ${prod_pt}
  []
  [pp_inj]
    type = PointValue
    variable = pp
    point = ${inj_pt}
  []
  [prod_bhp_Pa]
    type = FunctionValuePostprocessor
    function = prod_bhp
    point = ${prod_pt}
  []
  [pressure_drop_Pa]
    type = ParsedPostprocessor
    pp_names   = 'pp_inj prod_bhp_Pa'
    pp_symbols = 'pinj pbh'
    expression = 'pinj-pbh'
  []
  [pressure_drop_MPa]
    type = ParsedPostprocessor
    pp_names   = 'pressure_drop_Pa'
    pp_symbols = 'dp'
    expression = 'dp/1e6'
  []
  [T_reservoir_avg]
    type = ElementAverageValue
    variable = T
    block = 'block'
  []
  [reservoir_cooling_K]
    type = ParsedPostprocessor
    pp_names   = 'T_reservoir_avg'
    pp_symbols = 'Tavg'
    expression = '${T_ref} - Tavg'
  []
[]

[FluidProperties]
  [Water97FluidProperties]
    type = Water97FluidProperties
  []
[]

[Materials]
  [solid_bulk_density_for_gravity]
    type = GenericConstantMaterial
    prop_names = density
    prop_values = 2772.68093782
    block = 'cap bed block'
  []
  [porosity_cap]
    type = PorousFlowPorosityLinear
    porosity_ref = ${phi_cap_bed}
    block = 'cap bed'
    P_ref = insitu_pp
    P_coeff = ${P_coeff_cap_bed}
    T_ref = T_stress_free
    T_coeff = ${T_coeff_cap_bed}
    epv_ref = ${epv_ref}
    epv_coeff = ${epv_coeff_cap_bed}
  []
  [perm_cap]
    type = PorousFlowPermeabilityConst
    permeability = '${k_cap_bed} 0 0   0 ${k_cap_bed} 0   0 0 ${k_cap_bed}'
    block = 'cap bed'
  []
  [internal_energy_cap]
    type = PorousFlowMatrixInternalEnergy
    density = ${rho_s_cap}
    specific_heat_capacity = ${cp_s_cap}
    block = 'cap bed'
  []
  [k_cap]
    type = PorousFlowThermalConductivityIdeal
    dry_thermal_conductivity = '${kth_cap_bed} 0 0  0 ${kth_cap_bed} 0  0 0 ${kth_cap_bed}'
    block = 'cap bed'
  []
  [porosity_block]
    type = PorousFlowPorosityLinear
    porosity_ref = ${phi_block}
    block = 'block'
    P_ref = insitu_pp
    P_coeff = ${P_coeff}
    T_ref = T_stress_free
    T_coeff = ${T_coeff}
    epv_ref = ${epv_ref}
    epv_coeff = ${epv_coeff}
  []
  [perm_block]
    type = PorousFlowPermeabilityKozenyCarmanFromVar
    A = perm_block_A

    m = ${kc_m_block}
    n = ${kc_n_block}
    block = 'block'
  []
  [internal_energy_block]
    type = PorousFlowMatrixInternalEnergy
    density = ${rho_s_block}
    specific_heat_capacity = ${cp_s_block}
    block = 'block'
  []
  [k_block]
    type = PorousFlowThermalConductivityIdeal
    dry_thermal_conductivity = '${kth_block} 0 0  0 ${kth_block} 0  0 0 ${kth_block}'
    block = 'block'
  []
  [strain]
    type = ComputeSmallStrain
    displacements = 'disp_x disp_y'
    block = 'cap bed block'
    eigenstrain_names = 'ini_stress thermal_contribution'
  []
  [thermal_eigenstrain]
    type = ComputeThermalExpansionEigenstrain
    temperature = T
    stress_free_temperature = T_stress_free
    thermal_expansion_coeff = ${alphaT_block}
    block = 'cap bed block'
    eigenstrain_name = thermal_contribution
  []
  [elasticity_tensor_block]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${E_block}
    poissons_ratio = ${nu_block}
    block = 'block'
  []
  [elasticity_tensor_cap]
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${E_cap}
    poissons_ratio = ${nu_cap}
    block = 'cap bed'
  []
  [stress]
    type = ComputeLinearElasticStress
    block = 'cap bed block'
  []
  [ini_stress]
    type = ComputeEigenstrainFromInitialStress
    initial_stress = 'sigmaH_eff 0 0   0 sigmaV_eff 0   0 0 sigma3_eff'
    eigenstrain_name = ini_stress
    block = 'cap bed block'
  []
[]

[Preconditioning]
  [entire_jacobian]
    type = SMP
    full = true
    petsc_options_iname = '-pc_type -pc_factor_mat_solver_type'
    petsc_options_value = 'lu       mumps'
  []
[]

[Executioner]
  type = Transient
  solve_type = NEWTON
  line_search = 'none'
  automatic_scaling = true
  nl_abs_tol = ${nl_abs_tol}
  nl_rel_tol = ${nl_rel_tol}
  l_tol      = ${l_tol}
  l_max_its  = ${l_max_its}
  nl_max_its = ${nl_max_its}
  dt = ${dt0}
  [TimeStepper]
    type = IterationAdaptiveDT
    growth_factor = ${dt_growth}
    optimal_iterations = ${dt_opt_its}
    cutback_factor = ${dt_cutback}
    dt = ${dt0}
  []
  end_time = ${end_time}
  dtmax = 1.58e7
[]

[VectorPostprocessors]
  [point_sample1]
    type = PositionsFunctorValueSampler
    discontinuous = False
    functors  = 'pp T'
    positions = 'all_elements'
    sort_by   = 'x'
    execute_on = 'TIMESTEP_END'
  []
[]

[Positions]
  [all_elements]
    type = ElementCentroidPositions
  []
[]

[Outputs]
  print_linear_residuals = false
  [csv_out]
    type = CSV
    execute_on = 'INITIAL TIMESTEP_END FINAL'
    execute_vector_postprocessors_on = FINAL
  []
  [snap_out]
    type = Exodus
    show = 'T pp'
    time_step_interval = 3
    execute_on = 'TIMESTEP_END FINAL'
  []
[]

# ==========================
mesh_file = /home/yl3825/projects/Geothermal_Journal/THM_simulaiton/nf_slice_0005.msh
perm_block_file = '/home/yl3825/Pycharm_projects/Journal_Inverse_problem/Geothermal_generation/Journal paper/output_perm_moose_5/nf_slice_0000_mean_k.i'
prod_bh_file = production.bh
prod_pt = '0  250 0'
inj_T = 296
prod_p = 20e6
# ★2026-08-05:生产井完井段长度 [m]。2D 模型按单位厚度积分(注入用 q/Lz),故生产井
#   也必须是 1 m 段才与注入端同口径。旧 .bh 写 z=1/z=0 两点,但网格严格 2D(z≡0),
#   z=1 点被 MOOSE 静默丢弃 ⇒ 实际只有 0.5 m、井指数减半。现改为单点 + 本参数。
prod_line_length = 1.0
injection_mass_rate = 1
# Per-stage injection mass rate [kg/s] used in the 2D single-stage model.
# Assumed total number of stages: 20-30.
# If the total field injection rate is 30-50 kg/s, then the corresponding
# per-stage injection mass-rate range is:
#   minimum: 30 / 30 = 1.0 kg/s
#   maximum: 50 / 20 = 2.5 kg/s
# Therefore, a reasonable per-stage injection mass-rate range is 1.0-2.5 kg/s.








# ==========================
# Top-level user parameters
# ==========================
# Domain / reference points
xmin = -400 #✅
xmax =  400 #✅
ymin = -500 #✅
ymax =  500 #✅
ml_pt = '${xmin} 0 0' #✅
mr_pt = '${xmax} 0 0' #✅
mb_pt = '0 ${ymin} 0' #✅
mt_pt = '0 ${ymax} 0' #✅

inj_pt  = '0 -250 0'   # fixed and don't change #✅
Lz = 5
# Out-of-plane thickness of the stimulated DAMAGE ZONE represented by this 2D slice [m].
# Physical picture (2026-07-09): unlike studies that idealize the hydraulic fracture as a
# zero-thickness 2D plane, this model represents a finite-thickness damage zone of a FEW
# metres (crushed/stimulated rock around the main fracture). All 2D<->3D scalings use this
# thickness: injected mass per unit thickness = injection_mass_rate/Lz, produced mass and
# heat are multiplied back by Lz in the postprocessors.
# NOTE: with the same per-stage rate (1 kg/s), a thinner slab means a ~(50/Lz)x higher
# per-thickness forcing than the legacy Lz=50 assumption -> higher pressure drop and faster
# thermal breakthrough. Check injection pressure stays below Shmin (~70 MPa at 3 km).

# Initial conditions / gradients
pp_ic = 30e6 #✅
T_ic  = 354.15 #✅
y_ref  = 0.0 #✅
pp_ref = ${pp_ic} #✅
T_ref  = ${T_ic} #✅
epv_ref = 0 #✅
# In-situ stress / pressure / temperature gradients (CUBO-based, linearized)

sigmaV_insitu_ref = 78.6e6   # Pa #✅
# Reference vertical in-situ total stress at ~3 km depth.
# Estimated from the CUBO stress profile by linearly extrapolating Sv to 3000 m.
# Reported CUBO values:
#   Sv = 62.4 MPa at 2403.4 m
#   Sv = 74.6 MPa at 2852.0 m
# Therefore, Sv at ~3000 m is approximated as 78.6 MPa.

sigmaH_insitu_ref = 103.4e6  # Pa #✅
# Reference maximum horizontal in-situ total stress at ~3 km depth.
# Estimated from the CUBO stress profile by linearly extrapolating SHmax to 3000 m.
# Reported CUBO values:
#   SHmax = 74.3 MPa at 2403.4 m
#   SHmax = 96.2 MPa at 2852.0 m
# Therefore, SHmax at ~3000 m is approximated as 103.4 MPa.

sigma3_insitu_ref = 70.7e6   # Pa #✅
# Reference minimum horizontal in-situ total stress at ~3 km depth.
# Estimated from the CUBO stress profile by linearly extrapolating Shmin to 3000 m.
# Reported CUBO values:
#   Shmin = 54.9 MPa at 2403.4 m
#   Shmin = 66.8 MPa at 2852.0 m
# Therefore, Shmin at ~3000 m is approximated as 70.7 MPa.

dpdy = -9810                 # Pa/m #✅
# Pore-pressure gradient with respect to y.
# Negative because the model uses y increasing upward, so pressure increases with depth.
# This is approximately hydrostatic for water:
#   dp/dz ≈ rho*g ≈ 1000 * 9.81 = 9810 Pa/m

dTdy = -0.023                # K/m #✅
# Geothermal temperature gradient with respect to y.
# Negative because temperature increases with depth while y increases upward.
# This corresponds to about 23 K/km.

dsigmaV_insitu_dy = -2.72e4  # Pa/m #✅
# Vertical stress gradient with respect to y.
# Estimated from CUBO data assuming Sv varies linearly with depth:
#   (74.6 - 62.4) MPa / (2852.0 - 2403.4) m ≈ 0.0272 MPa/m
#   = 2.72e4 Pa/m
# Negative sign is used because stress increases downward while y increases upward.

dsigmaH_insitu_dy = -4.88e4  # Pa/m #✅
# Maximum horizontal stress gradient with respect to y.
# Estimated from CUBO data assuming SHmax varies linearly with depth:
#   (96.2 - 74.3) MPa / (2852.0 - 2403.4) m ≈ 0.0488 MPa/m
#   = 4.88e4 Pa/m
# Negative sign is used because stress increases downward while y increases upward.

dsigma3_insitu_dy = -2.65e4  # Pa/m #✅
# Minimum horizontal stress gradient with respect to y.
# Estimated from CUBO data assuming Shmin varies linearly with depth:
#   (66.8 - 54.9) MPa / (2852.0 - 2403.4) m ≈ 0.0265 MPa/m
#   = 2.65e4 Pa/m
# Negative sign is used because stress increases downward while y increases upward.


# Rock / porous properties
phi_block = 3.9e-2 # from the paper of Koenraad Beckers #✅
# Cap rock / bedrock assumptions adopted from the paper
# "Numerical simulation of heat extraction performance in enhanced geothermal system with multilateral wells":
# porosity = 0.01 #✅
# permeability = 1e-18 m^2 #✅
phi_cap_bed = 1e-3 #✅
k_cap_bed = 1e-18 #✅
kth_cap_bed = 2.8 #✅
rho_s_block = 2800 # from the paper of Koenraad Beckers #✅
cp_s_block  = 1000 # from the paper of Koenraad Beckers #✅
kth_block   = 2.8 # from the paper of Koenraad Beckers #✅
epv_coeff = 0.541 # alpha - phi_ref for current-volume porosity #✅
T_coeff = -1.623e-5 # Calculated from the linear thermo-poroelasticity theory (Coussy, 2004) #✅
P_coeff = 7.9160516129e-12 # (alpha - phi_ref) / K_s #✅
epv_coeff_cap_bed = 0.579 # alpha - phi_ref for current-volume porosity
T_coeff_cap_bed = -1.737e-5 # -3 alpha_T^s (alpha - phi_ref)
P_coeff_cap_bed = 8.47207741935e-12 # (alpha - phi_ref) / K_s
rho_s_cap = ${rho_s_block} #✅
cp_s_cap  = ${cp_s_block} #✅

# Mechanical part
E_block  = 31e9 # from the paper of Koenraad Beckers #✅
nu_block = 0.32 # from the paper of Koenraad Beckers #✅
alphaT_block = 1.0e-5 # # from the paper of Koenraad Beckers #✅
E_cap  = ${E_block} #✅
nu_cap = ${nu_block} #✅
biot = 0.58 # from the paper of Koenraad Beckers #✅




# --------------------------
# "Magic numbers" pulled up
# --------------------------
nodeset_tolerance = 1e-3
prod_bhp_ramp_t_end = 2000        # seconds (PiecewiseLinear x end)

MW = 1e6 #✅
GJ = 1e9 #✅
# Kozeny-Carman exponents
kc_m_block = 2 #✅
kc_n_block = 3 #✅

# Peaceman borehole "character" parameter
peaceman_character = 1

# Executioner / timestep controls
dt0 = 200
dt_growth = 1.5
dt_cutback = 0.7
dt_opt_its = 12
end_time = 3.16e8

nl_abs_tol = 1e-5
nl_rel_tol = 1e-5
l_tol = 1e-6
l_max_its = 1000
nl_max_its = 50

# ==========================
# MOOSE input
# ==========================

[Mesh]
  final_generator = mt_nodeset
  # Natural-fracture traces remain in the static equivalent-permeability field.
  # Do not generate a lower-dimensional fracture block: that would duplicate
  # their hydraulic effect and add an uncalibrated line-continuum mechanics model.
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
  [pp_ic_gradient] # correct, used the value near 3km deep ✅
    type = FunctionIC
    variable = pp
    function = pp_init
  []
  [T_ic_gradient] # correct, used the value near 3km deep ✅
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
  [perm_block_fn] # ✅
    type = PiecewiseMultilinear
    data_file = ${perm_block_file}
  []
  [prod_bhp] # Ramp from the local hydrostatic pressure at the producer to prod_p.
    type = ParsedFunction
    expression = 'if(t < ${prod_bhp_ramp_t_end}, (${pp_ref}+${dpdy}*(y-${y_ref}))+(${prod_p}-(${pp_ref}+${dpdy}*(y-${y_ref})))*t/${prod_bhp_ramp_t_end}, ${prod_p})'
  []
  [pp_init] # ✅
    type = ParsedFunction
    expression = '${pp_ref} + ${dpdy}*(y - ${y_ref})'
  []
  [T_init] # ✅
    type = ParsedFunction
    expression = '${T_ref} + ${dTdy}*(y - ${y_ref})'
  []
  [injection_rate] # ✅
    type = ParsedFunction
    expression = '${injection_mass_rate}/${Lz}'
  []
  [sigmaV_insitu_init] # total vertical in-situ stress ✅
    type = ParsedFunction
    expression = '-(${sigmaV_insitu_ref} + ${dsigmaV_insitu_dy}*(y - ${y_ref}))'
  []
  [sigmaH_insitu_init] # total maximum horizontal in-situ stress ✅
    type = ParsedFunction
    expression = '-(${sigmaH_insitu_ref} + ${dsigmaH_insitu_dy}*(y - ${y_ref}))'
  []
  [sigmaV_eff] # effective vertical stress ✅
    type = ParsedFunction
    expression = '-(${sigmaV_insitu_ref} + ${dsigmaV_insitu_dy}*(y - ${y_ref})) + ${biot}*(${pp_ref} + ${dpdy}*(y - ${y_ref}))'
  []
  [sigmaH_eff] # effective maximum horizontal stress ✅
    type = ParsedFunction
    expression = '-(${sigmaH_insitu_ref} + ${dsigmaH_insitu_dy}*(y - ${y_ref})) + ${biot}*(${pp_ref} + ${dpdy}*(y - ${y_ref}))'
  []
  [pressV_mag] # total vertical stress magnitude for traction BC ✅
    type = ParsedFunction
    expression = 'abs(-(${sigmaV_insitu_ref} + ${dsigmaV_insitu_dy}*(y - ${y_ref})))'
  []
  [pressH_mag] # total maximum horizontal stress magnitude for traction BC ✅
    type = ParsedFunction
    expression = 'abs(-(${sigmaH_insitu_ref} + ${dsigmaH_insitu_dy}*(y - ${y_ref})))'
  []
  [sigma3_insitu_init] # total minimum horizontal in-situ stress ✅
    type = ParsedFunction
    expression = '-(${sigma3_insitu_ref} + ${dsigma3_insitu_dy}*(y - ${y_ref}))'
  []
  [sigma3_eff] # effective minimum horizontal stress ✅
    type = ParsedFunction
    expression = '-(${sigma3_insitu_ref} + ${dsigma3_insitu_dy}*(y - ${y_ref})) + ${biot}*(${pp_ref} + ${dpdy}*(y - ${y_ref}))'
  []
[]


[PorousFlowFullySaturated] #✅
  coupling_type = ThermoHydroMechanical #✅
  gravity = '0 -9.81 0'  # fix2: y 是竖直方向，面内重力；默认(0,0,-10)在2D面外、等效无重力
  biot_coefficient = ${biot} #✅
  # Explicit stress AuxKernels below retain INITIAL output; avoid duplicate
  # action-generated stress kernels and unused Darcy-velocity variables.
  add_stress_aux = false
  add_darcy_aux = false
  porepressure = pp #✅
  temperature = T #✅
  displacements = 'disp_x disp_y' #✅
  fp = Water97FluidProperties #✅
  block = 'cap bed block'#✅
  stabilization = full #✅
  eigenstrain_names = 'thermal_contribution' #✅
  fluid_properties_type = 'PorousFlowSingleComponentFluid' #✅
  multiply_by_density = true #✅
[]


[BCs]
  [far_r] #✅
    type = Pressure
    boundary = r
    variable = disp_x
    function = pressH_mag
  []
  [far_t] #✅
    type = Pressure
    boundary = t
    variable = disp_y
    function = pressV_mag
  []
  [far_l] #✅
    type = Pressure
    boundary = l
    variable = disp_x
    function = pressH_mag
  []
  [pin_bottom_x] # remove horizontal rigid-body motion without clamping the whole base
    type = DirichletBC
    boundary = mb
    variable = disp_x
    value = 0
  []
  [roller_bottom_y] # vertical support; horizontal displacement remains free along the base
    type = DirichletBC
    boundary = b
    variable = disp_y
    value = 0
  []
[]

[UserObjects]
  [produced_mass] #✅
    type = PorousFlowSumQuantity
  []
  [produced_heat] #✅
    type = PorousFlowSumQuantity
  []
[]

[DiracKernels]
  [inject_fluid] #✅
    type = PorousFlowPointSourceFromPostprocessor #✅
    mass_flux = inj_mass_flux_pp  #✅
    point = ${inj_pt} #✅
    variable = pp #✅
  []
  [source_h] #✅
    type = PorousFlowPointEnthalpySourceFromPostprocessor #✅
    variable = T #✅
    mass_flux = inj_mass_flux_pp #✅
    point = ${inj_pt}  #✅
    T_in = ${inj_T} #✅
    pressure = pp #✅
    fp = Water97FluidProperties #✅
  []
  [fluid_production] #✅
    type = PorousFlowPeacemanBorehole #✅
    variable = pp #✅
    SumQuantityUO = produced_mass #✅
    point_file = ${prod_bh_file} #✅
    function_of = pressure #✅
    fluid_phase = 0 #✅
    bottom_p_or_t = prod_bhp #✅
    unit_weight = '0 0 0' #✅
    use_mobility = true #✅
    line_length = ${prod_line_length} # 2026-08-05: 2D 网格必须显式给完井段长(见 prod_line_length)
    character = ${peaceman_character} #✅
  []
  [remove_heat_at_production_well] #✅
    type = PorousFlowPeacemanBorehole #✅
    variable = T #✅
    SumQuantityUO = produced_heat #✅
    point_file = ${prod_bh_file} #✅
    function_of = pressure #✅
    fluid_phase = 0 #✅
    bottom_p_or_t = prod_bhp #✅
    unit_weight = '0 0 0' #✅
    use_mobility = true #✅
    use_enthalpy = true #✅
    line_length = ${prod_line_length} # 与质量井保持同一完井段
    character = ${peaceman_character} #✅
  []
[]

[Postprocessors]
  [prod_mass_kg_this_step] #✅
    type = PorousFlowPlotQuantity
    uo = produced_mass
  []
  [heat_J_this_step] #✅
    type = PorousFlowPlotQuantity
    uo = produced_heat
  []
  [dt_pp] #✅
    type = TimestepSize
  []
  [m_prod_kgs] #✅
    type = ParsedPostprocessor
    pp_names   = 'prod_mass_kg_this_step dt_pp'
    pp_symbols = 'm dt'
    expression = '(m/dt)*${Lz}'
  []
  [inj_mass_flux_pp] #✅
    type = FunctionValuePostprocessor
    function = injection_rate
    execute_on = timestep_begin
  []
  [m_inj_kgs] #✅
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
    # Useful produced thermal power relative to the injected-fluid enthalpy.
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
  [cum_heat_J] #✅
    type = CumulativeValuePostprocessor
    postprocessor = net_heat_J_this_step
  []
  [cum_heat_GJ] #✅
    type = ParsedPostprocessor
    pp_names   = 'cum_heat_J'
    pp_symbols = 'E'
    expression = 'E/${GJ}'
  []
  [T_prod] #✅
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
  [solid_bulk_density_for_gravity]  # fix2: action 自动加的 Gravity 体力核需要 density 属性
    type = GenericConstantMaterial
    prop_names = density
    prop_values = 2772.68093782  # = -dsigmaV_insitu_dy/g，与垂向应力梯度平衡
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
    type = PorousFlowPermeabilityConst # ✅
    permeability = '${k_cap_bed} 0 0   0 ${k_cap_bed} 0   0 0 ${k_cap_bed}' # ✅
    block = 'cap bed' # ✅
  []
  [internal_energy_cap] # ✅
    type = PorousFlowMatrixInternalEnergy # ✅
    density = ${rho_s_cap} # ✅
    specific_heat_capacity = ${cp_s_cap} # ✅
    block = 'cap bed' # ✅
  []
  [k_cap] # ✅
    type = PorousFlowThermalConductivityIdeal # ✅
    dry_thermal_conductivity = '${kth_cap_bed} 0 0  0 ${kth_cap_bed} 0  0 0 ${kth_cap_bed}' # ✅
    block = 'cap bed' # ✅
  []
  [porosity_block] # ✅
    type = PorousFlowPorosityLinear
    porosity_ref = ${phi_block}
    block = 'block'
    P_ref = insitu_pp # Spatially varying pre-injection pressure
    P_coeff = ${P_coeff} # calculated
    T_ref = T_stress_free # Spatially varying pre-injection temperature
    T_coeff = ${T_coeff} # calculated
    epv_ref = ${epv_ref} # Equal to 0
    epv_coeff = ${epv_coeff} # calculated
  []
  [perm_block] # ✅
    type = PorousFlowPermeabilityKozenyCarmanFromVar # ✅
    A = perm_block_A # ✅
    # poroperm_function = kozeny_carman_phi0 # ✅
    m = ${kc_m_block} # ✅
    n = ${kc_n_block} # ✅
    block = 'block' # ✅
  []
  [internal_energy_block] # ✅
    type = PorousFlowMatrixInternalEnergy # ✅
    density = ${rho_s_block} # ✅
    specific_heat_capacity = ${cp_s_block} # ✅
    block = 'block' # ✅
  []
  [k_block] # ✅
    type = PorousFlowThermalConductivityIdeal # ✅
    dry_thermal_conductivity = '${kth_block} 0 0  0 ${kth_block} 0  0 0 ${kth_block}' # ✅
    block = 'block' # ✅
  []
  [strain]  # ✅
    type = ComputeSmallStrain
    displacements = 'disp_x disp_y'
    block = 'cap bed block'
    eigenstrain_names = 'ini_stress thermal_contribution'
  []
  [thermal_eigenstrain] # ✅
    type = ComputeThermalExpansionEigenstrain
    temperature = T
    stress_free_temperature = T_stress_free
    thermal_expansion_coeff = ${alphaT_block}
    block = 'cap bed block'
    eigenstrain_name = thermal_contribution
  []
  [elasticity_tensor_block] # ✅
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${E_block}
    poissons_ratio = ${nu_block}
    block = 'block'
  []
  [elasticity_tensor_cap] # ✅
    type = ComputeIsotropicElasticityTensor
    youngs_modulus = ${E_cap}
    poissons_ratio = ${nu_cap}
    block = 'cap bed'
  []
  [stress] # ✅
    type = ComputeLinearElasticStress
    block = 'cap bed block'
  []
  [ini_stress] #✅
    type = ComputeEigenstrainFromInitialStress #✅
    initial_stress = 'sigmaH_eff 0 0   0 sigmaV_eff 0   0 0 sigma3_eff' #✅
    eigenstrain_name = ini_stress #✅
    block = 'cap bed block' #✅
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
[]

[VectorPostprocessors]
  [point_sample1]
    type = PositionsFunctorValueSampler
    discontinuous = False
    functors  = 'pp T'
    positions = 'all_elements'
    sort_by   = 'x'
    execute_on = FINAL
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
    execute_on = FINAL
  []
[]

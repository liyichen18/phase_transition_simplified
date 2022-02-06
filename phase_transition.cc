/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2016 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Timo Heister, Clemson University, 2016
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>


#include <deal.II/lac/generic_linear_algebra.h>

//#define FORCE_USE_OF_TRILINOS
#define USE_DIRECT_SOLVER // direct solver cannot be used with block matrix

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <cmath>
#include <fstream>
#include <iostream>

namespace Step55
{
using namespace dealii;

  enum class TestCase
  {
    test1,
    test2
  };
  static const char *enum_str[] = {"test1", "test2"};

namespace InlineFunctions
{
  inline
  double r(const double phi)
  {
    if(phi<0.)
      return 0.;
    else if(phi>1.)
      return 1.;
    else
      return std::pow(phi, 3.) * (10. - 15. * phi + 6. * phi * phi);
  }

  inline
  double r_prime(const double phi)
  {
    if(phi<0. || phi >1.)
      return 0.;
    else
      return 30. * phi * phi * (1. - phi) * (1. - phi);
  }

  inline
  double r_prime_prime(const double phi)
  {
    if(phi<0. || phi >1.)
      return 0.;
    else
      return 60. * phi * (1. - phi) * (1. - 2. * phi);
  }

  // used in wall energy
  inline
  double q(const double phi)
  {
    return (1. - std::cos(numbers::PI * phi)) * 0.5;
  }

  inline
  double q_prime(const double phi)
  {
    return  std::sin(numbers::PI * phi) * 0.5 * numbers::PI;
  }

  // interpolation functions for 1/rho, c
  inline
  double f(const double phi, const double psi,
                    const double fl, const double fs, const double fg)
  {
    return  fl * r(phi) * r(psi) +
            fs * r(phi) * (1.-r(psi)) +
            fg * (1. - r(phi));
  }

  // partial f/ partial phi
  inline
  double f_partial_phi(const double phi, const double psi,
                    const double fl, const double fs, const double fg)
  {
    return  r_prime(phi) * (r(psi) * (fl - fs) + fs - fg);
  }

  // partial f/ partial psi
  inline
  double f_partial_psi(const double phi, const double psi,
                    const double fl, const double fs)
  {
    return  r(phi) * r_prime(psi) * (fl - fs);
  }

  // partial^2 f/ partial phi^2
  inline
  double f_partial2_phi2(const double phi, const double psi,
                    const double fl, const double fs, const double fg)
  {
    return  r_prime_prime(phi) * (r(psi) * (fl - fs) + fs - fg);
  }

  // partial^2 f/ partial psi^2
  inline
  double f_partial2_psi2(const double phi, const double psi,
                    const double fl, const double fs)
  {
    return  r(phi) * r_prime_prime(psi) * (fl - fs);
  }

  // partial f/ (partial phi partial psi)
  inline
  double f_partial_phi_partial_psi(const double phi, const double psi,
                    const double fl, const double fs)
  {
    return  r_prime(phi) * r_prime(psi) * (fl - fs);
  }

  // interpolation function for eta

  inline
  double g(const double phi, const double psi,
                    const double gl, const double gs, const double gg)
  {
    return  gl * phi * psi + gs * phi * (1.-psi) + gg * (1. - phi);
  }

  inline
  double g_pratial_phi(const double psi,
                    const double gl, const double gs, const double gg)
  {
    return  psi * (gl - gs) + gs - gg;
  }

  inline
  double g_pratial_psi(const double phi,
                    const double gl, const double gs)
  {
    return  phi * (gl - gs);
  }

  // double-well potential

  inline
  double w(const double phi, const double eps)
  {
    return  phi * phi * (1. - phi) * (1. - phi) / (eps*eps);
  }

  inline
  double w_prime(const double phi, const double eps)
  {
    return  2. * phi * (1. - phi) * (1. - 2.*phi) / (eps*eps);
  }

  inline
  double w_prime_prime(const double phi, const double eps)
  {
    return  (2.  - 12. * phi + 12. * phi * phi) / (eps*eps);
  }

} //namespace inline funcitons

  template <int dim>
  struct ComponentIndices
  {
    const unsigned int       velocities = 0;
    const unsigned int       pressure = dim;
    const unsigned int       temperature = dim + 1;
    const unsigned int       phi_ch = dim + 2;   //CH
    const unsigned int       mu_phi_ch = dim + 3;
    const unsigned int       psi_ac = dim + 4;   //AC
    const unsigned int       mu_psi_ac = dim + 5;
  };

  template <int dim>
  struct Extractors
  {
    Extractors (const ComponentIndices<dim> &component_indices)
      : velocities(component_indices.velocities)
      , pressure(component_indices.pressure)
      , temperature((component_indices.temperature))
      , phi_ch(component_indices.phi_ch)
      , mu_phi_ch(component_indices.mu_phi_ch)
      , psi_ac(component_indices.psi_ac)
      , mu_psi_ac(component_indices.mu_psi_ac)
    {}

    Extractors (const Extractors<dim> &ex)
      : velocities(ex.velocities.first_vector_component)
      , pressure(ex.pressure.component)
      , temperature(ex.temperature.component)
      , phi_ch(ex.phi_ch.component)
      , mu_phi_ch(ex.mu_phi_ch.component)
      , psi_ac(ex.psi_ac.component)
      , mu_psi_ac(ex.mu_psi_ac.component)
    {}

    FEValuesExtractors::Vector              velocities;
    FEValuesExtractors::Scalar              pressure;
    FEValuesExtractors::Scalar              temperature;
    FEValuesExtractors::Scalar              phi_ch;   //CH
    FEValuesExtractors::Scalar              mu_phi_ch;
    FEValuesExtractors::Scalar              psi_ac;   //AC
    FEValuesExtractors::Scalar              mu_psi_ac;
  };

namespace InitialConditions
{


  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
      InitialValues (const double epsilon, const double initial_t, const TestCase testcase, const Extractors<dim> &ex)
                  : Function<dim>(dim + 6),
                    eps(epsilon),
                    initial_temperature(initial_t),
                    test_case(testcase),
                    extractors(ex)
                    {}

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &  value) const override;
  private:
      const double eps;
      const double initial_temperature;
      const TestCase test_case;
      const Extractors<dim> extractors;

  };

  template <int dim>
  void InitialValues<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &  values) const
  {
      const double eps1=eps * std::sqrt(2);  //sqrt(2)* eps
      const double x=p(0);
      const double y=p(1);

      switch (test_case)
        {
        case TestCase::test1: {
            Point<dim> center;
            const double R = 1.;
            double r;
            switch (dim)
              {
              case 2: //2D case
                {
                  center = Point<dim>(0,0);
                  r = std::sqrt(x*x + y*y);
                  break;
                }
              case 3: //3D case
                {
                  ExcNotImplemented("3D not implemented yet!");
                  break;
                }
              }

            const double d = r - R;
            const double psi = 0.5 * (1. + std::tanh(d/eps1));

            for(unsigned int comp = 0; comp < values.size(); ++comp)
              {
                if (comp == extractors.psi_ac.component)
                  values(comp) = psi;
                else if (comp == extractors.temperature.component)
                  values(comp) = initial_temperature;
                else if (comp == extractors.phi_ch.component)
                  values(comp) = 1.;
                else
                  values(comp) = 0;

              }
            break;
          }
        case TestCase::test2: {

            break;
          }
        default:
          Assert(false, ExcNotImplemented("Please choose the right test case"));
        }
  }

}// namespace initialcondition

namespace LinearSolvers
{
template <class Matrix, class Preconditioner>
class InverseMatrix : public Subscriptor
{
public:
    InverseMatrix(const Matrix &m, const Preconditioner &preconditioner);

    template <typename VectorType>
    void vmult(VectorType &dst, const VectorType &src) const;

private:
    const SmartPointer<const Matrix> matrix;
    const Preconditioner &           preconditioner;
};


template <class Matrix, class Preconditioner>
InverseMatrix<Matrix, Preconditioner>::InverseMatrix(
    const Matrix &        m,
    const Preconditioner &preconditioner)
    : matrix(&m)
    , preconditioner(preconditioner)
{}



template <class Matrix, class Preconditioner>
template <typename VectorType>
void
InverseMatrix<Matrix, Preconditioner>::vmult(VectorType &      dst,
        const VectorType &src) const
{
    SolverControl solver_control(src.size(), 1e-8 * src.l2_norm());
    SolverCG<LA::MPI::Vector> cg(solver_control);
    dst = 0;

    try
    {
        cg.solve(*matrix, dst, src, preconditioner);
    }
    catch (std::exception &e)
    {
        Assert(false, ExcMessage(e.what()));
    }
}


template <class PreconditionerA, class PreconditionerS>
class BlockDiagonalPreconditioner : public Subscriptor
{
public:
    BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                                const PreconditionerS &preconditioner_S);

    void vmult(LA::MPI::BlockVector &      dst,
               const LA::MPI::BlockVector &src) const;

private:
    const PreconditionerA &preconditioner_A;
    const PreconditionerS &preconditioner_S;
};

template <class PreconditionerA, class PreconditionerS>
BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::
BlockDiagonalPreconditioner(const PreconditionerA &preconditioner_A,
                            const PreconditionerS &preconditioner_S)
    : preconditioner_A(preconditioner_A)
    , preconditioner_S(preconditioner_S)
{}


template <class PreconditionerA, class PreconditionerS>
void BlockDiagonalPreconditioner<PreconditionerA, PreconditionerS>::vmult(
    LA::MPI::BlockVector &      dst,
    const LA::MPI::BlockVector &src) const
{
    preconditioner_A.vmult(dst.block(0), src.block(0));
    preconditioner_S.vmult(dst.block(1), src.block(1));
}

} // namespace LinearSolvers

template <int dim>
class StokesProblem
{
public:
    StokesProblem(unsigned int velocity_degree, const TestCase & testcase);

    void run();

private:
#ifdef USE_DIRECT_SOLVER
  using VectorType = LA::MPI::Vector;
  using MatrixType = LA::MPI::SparseMatrix;
#else
    using VectorType = LA::MPI::BlockVector;
    using MatrixType = LA::MPI::BlockSparseMatrix;
#endif
    void make_grid();
    void setup_system();
    void setup_block_system();
    void setup_initial_condition();
    void make_boundary_constraints();
    Table<2, DoFTools::Coupling> make_coupling();
    void assemble_system(const bool assemble_matrix = true);
    void newton_iteration();
    void map_dofs_to_component(
        const DoFHandler<dim> &    dof,
        std::vector<unsigned int> &global_index_to_component);
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    std::vector<const FiniteElement<dim> *>
    create_fe_list(const unsigned int velocity_degree);

    std::vector<unsigned int> create_fe_multiplicities();



    const unsigned int velocity_degree;
    const unsigned int quadrature_degree;

    const double       viscosity;
    MPI_Comm     mpi_communicator;

    FESystem<dim>                             fe;
    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>                           dof_handler;

    std::vector<IndexSet> block_owned_partitioning;
    std::vector<IndexSet> block_relevant_partitioning;

    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints_newton_update;
    AffineConstraints<double> constraints_boundary;

    MatrixType system_matrix;
    MatrixType preconditioner_matrix;
    VectorType       locally_relevant_solution; //u_n+1
    VectorType       old_solution; //u_n
    VectorType       old_old_solution; //u_n-1
    VectorType       current_solution; //u_*
    VectorType       system_rhs;
    VectorType       newton_update;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    const ComponentIndices<dim> component_ids;
    const Extractors<dim> extractors;

    const double eps;

    const TestCase test_case;

    const unsigned n_refinement;

    double theta = 1.;

    const double density_s, density_l, density_g;
    const double inv_density_s, inv_density_l, inv_density_g;
    double present_timestep, old_timestep;
    const double cl,cs,cg;
    const double eta_l,eta_s,eta_g;
    const double surface_tension;
    const double lambda_phi, lambda_psi;
    const double mobility_phi, mobility_psi;
    const double latent_heat, melting_t;
    const double thermal_conductivity;
    const double initial_temperature; //Ta

    const MappingQ<dim> mapping;

    double hmin;




};


template <int dim>
std::vector<const FiniteElement<dim> *>
StokesProblem<dim>::create_fe_list(
  const unsigned int velocity_degree)
{
  std::vector<const FiniteElement<dim> *> fe_list;

  fe_list.push_back(
    new FE_Q<dim>(velocity_degree)); //(dim) fluid velocity
  fe_list.push_back(new FE_Q<dim>(velocity_degree - 1)); //(1)   pressure P
  fe_list.push_back(new FE_Q<dim>(velocity_degree)); //temperature
  fe_list.push_back(new FE_Q<dim>(velocity_degree)); //phi_ch
  fe_list.push_back(new FE_Q<dim>(velocity_degree)); //mu_phi_ch
  fe_list.push_back(new FE_Q<dim>(velocity_degree)); //phi_ac
  fe_list.push_back(new FE_Q<dim>(velocity_degree)); //mu_phi_ch

  return fe_list;
}

template <int dim>
std::vector<unsigned int>
StokesProblem<dim>::create_fe_multiplicities()
{
  // correspond to fe list
  std::vector<unsigned int> multiplicities;
  multiplicities.push_back(dim);
  multiplicities.push_back(1);
  multiplicities.push_back(1);
  multiplicities.push_back(1);
  multiplicities.push_back(1);
  multiplicities.push_back(1);
  multiplicities.push_back(1);

  return multiplicities;
}

template <int dim>
StokesProblem<dim>::StokesProblem(unsigned int velocity_degree,
                                  const TestCase &testcase)
    : velocity_degree(velocity_degree)
    , quadrature_degree(2 * velocity_degree + 4)
    , viscosity(0.1)
    , mpi_communicator(MPI_COMM_WORLD)
    , fe(create_fe_list(velocity_degree),
         create_fe_multiplicities())
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                        Triangulation<dim>::smoothing_on_refinement |
                        Triangulation<dim>::smoothing_on_coarsening))
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , component_ids(ComponentIndices<dim>())
    , extractors(ComponentIndices<dim>())
    , eps(0.05)
    , test_case(testcase)
    , n_refinement(4)
    , density_s(1.)
    , density_l(1.)
    , density_g(0.01)
    , inv_density_s(1./density_s)
    , inv_density_l(1./density_l)
    , inv_density_g(1./density_g)
    , present_timestep(0.001), old_timestep(present_timestep)
    , cl(100.), cs(100.), cg(100.)
    , eta_l(1.), eta_s(100.), eta_g(0.01)
    , surface_tension(0.1)
    , lambda_phi(1.), lambda_psi(3.*std::sqrt(2.) * surface_tension * eps * inv_density_l)/*which density should be used?*/
    , mobility_phi(1e-4), mobility_psi(1e-4)
    , latent_heat(1.)
    , melting_t(1.)
    , thermal_conductivity(1.)
    , initial_temperature(0.5)
    , mapping(1)
{




}

template <int dim>
void StokesProblem<dim>::make_grid()
{
  switch  (test_case)
    {
    case TestCase::test1: {
//        If the colorize flag is true,
//        then the boundary_ids of the boundary faces are assigned,
//            such that the lower one in x-direction is 0, the upper one is 1.
//            The indicators for the surfaces in y-direction are 2 and 3,
//            the ones for z are 4 and 5.
        const bool   colorize = true;
        const double width = 2.;
        GridGenerator::hyper_cube(triangulation, 0., width, colorize);
        triangulation.refine_global(n_refinement);

        break;
      }
    case TestCase::test2: {

        break;
      }
    default:
      Assert(false, ExcNotImplemented("Setting up iniital grid: Please choose the right test case"));
    }

  hmin = GridTools::minimal_cell_diameter(triangulation, mapping)/std::sqrt(dim*1.);
  pcout<<" hmin = "<<hmin<<std::endl;
}

template <int dim>
void StokesProblem<dim>::setup_initial_condition()
{
  VectorType tmp_initial_sol;
#ifdef USE_DIRECT_SOLVER
  tmp_initial_sol.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);
#else
  tmp_initial_sol.reinit(block_owned_partitioning, mpi_communicator);
#endif

  VectorTools::interpolate(dof_handler,
                           InitialConditions::InitialValues<dim>(eps,
                                                                 initial_temperature,
                                                                 test_case,
                                                                 extractors),
                           tmp_initial_sol);




  locally_relevant_solution = tmp_initial_sol;
  old_solution = locally_relevant_solution;
  old_old_solution = locally_relevant_solution;
  current_solution = old_solution;
}

template <int dim>
void StokesProblem<dim>::make_boundary_constraints()
{
  constraints_boundary.reinit(locally_relevant_dofs);
  constraints_newton_update.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_boundary);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints_newton_update);

  switch  (test_case)
    {
    case TestCase::test1: {

        ComponentMask vel_v_masked(fe.n_components(), false);
        vel_v_masked.set(extractors.velocities.first_vector_component, true);
        {
          // x=0, boundary id=0, velocity (u,v,w)
          // u = 0
          const types::boundary_id bc_id = 0;
          ComponentMask vel_u_masked(fe.n_components(), false);
          vel_u_masked.set(extractors.velocities.first_vector_component, true);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(fe.n_components()),
                                                   constraints_boundary,
                                                   vel_u_masked);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(fe.n_components()),
                                                   constraints_newton_update,
                                                   vel_u_masked);
        }

        {
          // x=y, boundary id=2, velocity (u,v,w)
          // v = 0
          const types::boundary_id bc_id = 2;
          ComponentMask vel_v_masked(fe.n_components(), false);
          vel_v_masked.set(extractors.velocities.first_vector_component + 1, true);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(fe.n_components()),
                                                   constraints_boundary,
                                                   vel_v_masked);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(fe.n_components()),
                                                   constraints_newton_update,
                                                   vel_v_masked);
        }

        {
          // x=width, boundary id=1,
          // T = Ta
          const types::boundary_id bc_id = 1;
          ComponentMask temperature_masked(fe.n_components(), false);
          temperature_masked.set(extractors.temperature.component, true);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   bc_id,
                                                   Functions::ConstantFunction<dim>(initial_temperature, fe.n_components()),
                                                   constraints_boundary,
                                                   temperature_masked);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(fe.n_components()),
                                                   constraints_newton_update,
                                                   temperature_masked);
        }

        {
          // y=width, boundary id=3,
          // T = Ta
          const types::boundary_id bc_id = 3;
          ComponentMask temperature_masked(fe.n_components(), false);
          temperature_masked.set(extractors.temperature.component, true);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   bc_id,
                                                   Functions::ConstantFunction<dim>(initial_temperature, fe.n_components()),
                                                   constraints_boundary,
                                                   temperature_masked);
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(fe.n_components()),
                                                   constraints_newton_update,
                                                   temperature_masked);
        }



        break;
      }
    case TestCase::test2: {

        break;
      }
    default:
      Assert(false, ExcNotImplemented("Setting up constraints: Please choose the right test case"));
    }

  constraints_boundary.close();
  constraints_newton_update.close();


}

template <int dim>
Table<2, DoFTools::Coupling> StokesProblem<dim>::make_coupling()
{
  const unsigned int n_components = fe.n_components();
  Table<2, DoFTools::Coupling> coupling(n_components, n_components);
  for (unsigned int c = 0; c < n_components; ++c)
    for (unsigned int d = 0; d < n_components; ++d)
      if (c == component_ids.temperature
               && d != component_ids.pressure)
        coupling[c][d] = DoFTools::always;
      else if (c == component_ids.mu_phi_ch
               && (d == component_ids.mu_phi_ch
                   || d == component_ids.phi_ch
                   || d == component_ids.psi_ac
                   || d == component_ids.temperature
                   || d == component_ids.pressure))
        coupling[c][d] = DoFTools::always;
      else if (c == component_ids.mu_psi_ac
               && (d == component_ids.mu_psi_ac
                   || d == component_ids.phi_ch
                   || d == component_ids.psi_ac
                   || d == component_ids.temperature
                   || d == component_ids.pressure
                   || (d >= component_ids.velocities && d < component_ids.velocities+dim)))
        coupling[c][d] = DoFTools::always;
      else if (c == component_ids.phi_ch
               && (d == component_ids.phi_ch
                   || d == component_ids.mu_phi_ch
                   || d == component_ids.psi_ac
                   || (d >= component_ids.velocities && d < component_ids.velocities+dim)))
        coupling[c][d] = DoFTools::always;
      else if (c == component_ids.psi_ac
               && (d == component_ids.psi_ac
                   || d == component_ids.phi_ch
                   || d == component_ids.mu_psi_ac
                   || (d >= component_ids.velocities && d < component_ids.velocities+dim)))
        coupling[c][d] = DoFTools::always;
  //todo: navier stokes coupling
      else if ((c >= component_ids.velocities && c < component_ids.velocities + dim)
               && ((d >= component_ids.velocities && d < component_ids.velocities + dim)
                   || d == component_ids.pressure
                   || d == component_ids.phi_ch
                   || d == component_ids.psi_ac))
        coupling[c][d] = DoFTools::always;
      else if (c == component_ids.pressure
               && ((d >= component_ids.velocities && d < component_ids.velocities + dim)
                   || d == component_ids.phi_ch
                   || d == component_ids.psi_ac))
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::none;

  bool print_pattern = true;
  if (print_pattern)
    {
      // visualize coupling
      FullMatrix<double> cell_coupling_mat(n_components,
                                           n_components);
      for (unsigned int c = 0; c < n_components; ++c)
        {
          for (unsigned int d = 0; d < n_components; ++d)
            {
              if (coupling[c][d] == DoFTools::always)
                cell_coupling_mat(c, d) = 1;
            }
        }
      unsigned int mpi_rank    = Utilities::MPI::this_mpi_process(mpi_communicator);
      pcout<<" coupling pattern: "<<std::endl;
      if(mpi_rank == 0)
        cell_coupling_mat.print(std::cout);
    }

  return coupling;
}

template <int dim>
void StokesProblem<dim>::setup_block_system()
{
#ifndef USE_DIRECT_SOLVER
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);

    const unsigned int n_components = fe.n_components();

    std::vector<unsigned int> sub_blocks(n_components, 0);

    sub_blocks[dim] = 1;

    for(auto i=0u; i<n_components - (dim + 1); ++i)
      sub_blocks[i+dim+1] = i + 2;

    for(auto i : sub_blocks)
      pcout<<" block: "<<i<<std::endl;

    const unsigned int n_blocks = sub_blocks[n_components - 1] + 1;

    pcout<<" n blocks: "<<n_blocks<<std::endl;

    DoFRenumbering::component_wise(dof_handler, sub_blocks);

    const std::vector<types::global_dof_index> dofs_per_block =
        DoFTools::count_dofs_per_fe_block(dof_handler, sub_blocks);

    pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << " (";
    for(auto i:dofs_per_block)
      pcout<< i <<"  "  ;
    pcout << ")"<< std::endl;

    block_owned_partitioning.resize(n_blocks);

    unsigned int block_starting_id = 0;
    for(unsigned int i=0; i<n_blocks; ++i)
      {
        block_owned_partitioning[i] =
            dof_handler.locally_owned_dofs().get_view(block_starting_id,
                                                      block_starting_id + dofs_per_block[i]);
        block_starting_id += dofs_per_block[i];
        pcout<<" owned partitioning size ( "<<i<<" ): "<<block_owned_partitioning[i].size()<<std::endl;
      }

    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    block_relevant_partitioning.resize(n_blocks);

    block_starting_id = 0;
    for(unsigned int i=0; i<n_blocks; ++i)
      {
        block_relevant_partitioning[i] = locally_relevant_dofs.get_view(block_starting_id,
                                                                  block_starting_id + dofs_per_block[i]);
        block_starting_id += dofs_per_block[i];
//        pcout<<" relevant partitioning size ( "<<i<<" ): "<<block_relevant_partitioning[i].size()<<std::endl;
      }

    make_boundary_constraints();


    {
        system_matrix.clear();

        const Table<2, DoFTools::Coupling> coupling = make_coupling();

        BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

        DoFTools::make_sparsity_pattern(
            dof_handler, coupling, dsp, constraints_newton_update, false);

        SparsityTools::distribute_sparsity_pattern(
            dsp,
            dof_handler.locally_owned_dofs(),
            mpi_communicator,
            locally_relevant_dofs);

//        std::cout<<" rows size: "<<block_owned_partitioning.size()
//                << " n block rows: "<<dsp.n_block_rows()<<std::endl;
        system_matrix.reinit(block_owned_partitioning, dsp, mpi_communicator);
    }

    locally_relevant_solution.reinit(block_owned_partitioning,
                                     block_relevant_partitioning,
                                     mpi_communicator);
    old_solution.reinit(block_owned_partitioning,
                        block_relevant_partitioning,
                        mpi_communicator);
    old_old_solution.reinit(old_solution);
    current_solution.reinit(block_owned_partitioning,
                        block_relevant_partitioning,
                        mpi_communicator);
    system_rhs.reinit(block_owned_partitioning, mpi_communicator);
    newton_update.reinit(block_owned_partitioning, mpi_communicator);
#endif
}


template <int dim>
void StokesProblem<dim>::setup_system()
{
#ifdef USE_DIRECT_SOLVER
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);
    pcout<<" dofs: "<<dof_handler.n_dofs()<<std::endl;

//    const unsigned int n_components = fe.n_components();

//    std::vector<unsigned int> sub_blocks(n_components, 0);

//    sub_blocks[dim] = 1;

//    for(auto i=0u; i<n_components - (dim + 1); ++i)
//      sub_blocks[i+dim+1] = i + 2;

//    for(auto i : sub_blocks)
//      pcout<<" block: "<<i<<std::endl;

//    const unsigned int n_blocks = sub_blocks[n_components - 1] + 1;

//    pcout<<" n blocks: "<<n_blocks<<std::endl;

//    DoFRenumbering::component_wise(dof_handler, sub_blocks);

    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();

    make_boundary_constraints();


    {
        system_matrix.clear();

        const Table<2, DoFTools::Coupling> coupling = make_coupling();

        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

        DoFTools::make_sparsity_pattern(
            dof_handler, coupling, dsp, constraints_newton_update, false);

        SparsityTools::distribute_sparsity_pattern(
            dsp,
            locally_owned_dofs,
            mpi_communicator,
            locally_relevant_dofs);

//        std::cout<<" rows size: "<<block_owned_partitioning.size()
//                << " n block rows: "<<dsp.n_block_rows()<<std::endl;
        system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
    }

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    old_solution.reinit(locally_relevant_solution);
    old_old_solution.reinit(old_solution);
    current_solution.reinit(locally_relevant_solution);

    system_rhs.reinit(locally_owned_dofs, mpi_communicator);
    newton_update.reinit(system_rhs);
#endif
}


template <int dim>
void StokesProblem<dim>::assemble_system(const bool assemble_matrix)
{
    TimerOutput::Scope t(computing_timer, "assembly");

    system_matrix         = 0;
//    preconditioner_matrix = 0;
    system_rhs            = 0;

    SymmetricTensor<2, dim> id_tensor;
    for (unsigned int d = 0; d < dim; ++d)
      id_tensor[d][d] = 1.;

    const double         ext_c2 = theta * present_timestep / old_timestep;
    const double         ext_c1 = ext_c2 + 1;

    // todo: check if it's accurate enough
    const QGauss<dim> quadrature_formula(quadrature_degree);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
//    FullMatrix<double> cell_matrix2(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<double>         phi_ch_star(n_q_points);
    std::vector<double>         phi_ch_n(n_q_points);
    std::vector<double>         psi_ac_star(n_q_points);
    std::vector<double>         psi_ac_n(n_q_points);
    std::vector<double>         mu_phi_ch_star(n_q_points);
    std::vector<double>         mu_psi_ac_star(n_q_points);
    std::vector<double>         mu_psi_ac_n(n_q_points);

    std::vector<Tensor<1,dim>>  grad_phi_ch_star(n_q_points);
    std::vector<Tensor<1,dim>>  grad_phi_ch_n(n_q_points);
    std::vector<Tensor<1,dim>>  grad_psi_ac_star(n_q_points);
    std::vector<Tensor<1,dim>>  grad_psi_ac_n(n_q_points);
    std::vector<Tensor<1,dim>>  grad_mu_phi_ch_star(n_q_points);
    std::vector<Tensor<1,dim>>  grad_mu_psi_ac_star(n_q_points);

    std::vector<Tensor<1,dim>>  vel_star(n_q_points);
    std::vector<Tensor<1,dim>>  vel_n(n_q_points);
    std::vector<Tensor<1,dim>>  vel_n_minus_1(n_q_points);
    std::vector<Tensor<2,dim>>  grad_vel_star(n_q_points);
    std::vector<Tensor<2,dim>>  grad_vel_n(n_q_points);
    std::vector<Tensor<2,dim>>  grad_vel_n_minus_1(n_q_points);

    std::vector<double>         temperature_star(n_q_points);
    std::vector<double>         temperature_n(n_q_points);
    std::vector<Tensor<1,dim>>  grad_temperature_star(n_q_points);
    std::vector<Tensor<1,dim>>  grad_temperature_n(n_q_points);

    std::vector<double>         pressure_star(n_q_points);

    // shape functions:
    std::vector<double> shape_phi_ch(dofs_per_cell);
    std::vector<double> shape_psi_ac(dofs_per_cell);
    std::vector<double> shape_phi_ch_theta(dofs_per_cell);
    std::vector<double> shape_psi_ac_theta(dofs_per_cell);
    std::vector<double> shape_mu_phi_ch(dofs_per_cell);
    std::vector<double> shape_mu_psi_ac(dofs_per_cell);

    std::vector<Tensor<1,dim>> grad_shape_phi_ch(dofs_per_cell);
    std::vector<Tensor<1,dim>> grad_shape_phi_ch_theta(dofs_per_cell);
    std::vector<Tensor<1,dim>> grad_shape_psi_ac(dofs_per_cell);
    std::vector<Tensor<1,dim>> grad_shape_psi_ac_theta(dofs_per_cell);
    std::vector<Tensor<1,dim>> grad_shape_mu_phi_ch(dofs_per_cell);
    std::vector<Tensor<1,dim>> grad_shape_mu_psi_ac(dofs_per_cell);

    std::vector<double> shape_inv_rho_theta(dofs_per_cell);
    std::vector<double> shape_inv_rho_partial_phi_theta(dofs_per_cell);
    std::vector<double> shape_inv_rho_partial_psi_theta(dofs_per_cell);
    std::vector<double> shape_rho_theta(dofs_per_cell);

    // delta H_theta_star
    std::vector<double> shape_h_ts(dofs_per_cell);
    // delta grad_inv_rho_theta
    std::vector<Tensor<1,dim>> grad_shape_inv_rho_theta(dofs_per_cell);

    std::vector<Tensor<2, dim>> grad_shape_vel(dofs_per_cell);
    std::vector<Tensor<1, dim>> shape_vel(dofs_per_cell);
    std::vector<double>         shape_div_vel(dofs_per_cell);
    std::vector<double>         shape_pressure(dofs_per_cell);
    std::vector<Tensor<1, dim>> shape_vel_theta(dofs_per_cell);

    std::vector<double>         shape_c_theta(dofs_per_cell);
    std::vector<double>         shape_c_partial_phi_theta(dofs_per_cell);
    std::vector<double>         shape_c_partial_psi_theta(dofs_per_cell);

    std::vector<double>         shape_eta_theta(dofs_per_cell);

    std::vector<Tensor<2,dim>>  shape_e_theta(dofs_per_cell);

    std::vector<Tensor<2,dim>>  shape_gamma_theta(dofs_per_cell);

    std::vector<double>         shape_temperature(dofs_per_cell);
    std::vector<double>         shape_temperature_theta(dofs_per_cell);
    std::vector<Tensor<1,dim>>  grad_shape_temperature(dofs_per_cell);
    std::vector<Tensor<1,dim>>  grad_shape_temperature_theta(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    double mat, rhs;
    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell->get_dof_indices(local_dof_indices);
            cell_matrix  = 0;
//            cell_matrix2 = 0;
            cell_rhs     = 0;

            fe_values.reinit(cell);
            fe_values[extractors.phi_ch].get_function_values(current_solution, phi_ch_star);
            fe_values[extractors.phi_ch].get_function_values(old_solution, phi_ch_n);
            fe_values[extractors.phi_ch].get_function_gradients(current_solution, grad_phi_ch_star);
            fe_values[extractors.phi_ch].get_function_gradients(old_solution, grad_phi_ch_n);

            fe_values[extractors.psi_ac].get_function_values(current_solution, psi_ac_star);
            fe_values[extractors.psi_ac].get_function_values(old_solution, psi_ac_n);
            fe_values[extractors.psi_ac].get_function_gradients(current_solution, grad_psi_ac_star);
            fe_values[extractors.psi_ac].get_function_gradients(old_solution, grad_psi_ac_n);

            fe_values[extractors.mu_phi_ch].get_function_values(current_solution, mu_phi_ch_star);
            fe_values[extractors.mu_phi_ch].get_function_gradients(current_solution, grad_mu_phi_ch_star);

            fe_values[extractors.mu_psi_ac].get_function_gradients(current_solution, grad_mu_psi_ac_star);

            fe_values[extractors.velocities].get_function_values(current_solution, vel_star);
            fe_values[extractors.velocities].get_function_values(old_solution, vel_n);
            fe_values[extractors.velocities].get_function_values(old_old_solution, vel_n_minus_1);
            fe_values[extractors.velocities].get_function_gradients(current_solution, grad_vel_star);
            fe_values[extractors.velocities].get_function_gradients(old_solution, grad_vel_n);
            fe_values[extractors.velocities].get_function_gradients(old_old_solution, grad_vel_n_minus_1);

            fe_values[extractors.temperature].get_function_values(current_solution, temperature_star);
            fe_values[extractors.temperature].get_function_values(old_solution, temperature_n);
            fe_values[extractors.temperature].get_function_gradients(current_solution, grad_temperature_star);
            fe_values[extractors.temperature].get_function_gradients(old_solution, grad_temperature_n);

            fe_values[extractors.pressure].get_function_values(current_solution, pressure_star);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
              // ts: theta star
              const double c1           = 1.-theta;
              const double phi_ch_ts    = theta * phi_ch_star[q] + c1 * phi_ch_n[q];
              const double psi_ac_ts    = theta * psi_ac_star[q] + c1 * psi_ac_n[q];

              const Tensor<1,dim> grad_phi_ch_ts    = theta * grad_phi_ch_star[q] + c1 * grad_phi_ch_n[q];
              const Tensor<1,dim> grad_psi_ac_ts    = theta * grad_psi_ac_star[q] + c1 * grad_psi_ac_n[q];

              const Tensor<1,dim> vel_ts         = theta * vel_star[q] + c1 * vel_n[q];
              const Tensor<2,dim> grad_vel_ts    = theta * grad_vel_star[q] + c1 * grad_vel_n[q];
              const double div_vel_ts            = trace(grad_vel_ts);
              const Tensor<1, dim> vel_bar       = ext_c1 * vel_n[q] - ext_c2 * vel_n_minus_1[q];
              const double div_vel_bar           = ext_c1 * trace(grad_vel_n[q]) - ext_c2 * trace(grad_vel_n_minus_1[q]);
              const double temperature_ts        = theta * temperature_star[q] + c1 * temperature_n[q];

              const Tensor<1,dim> grad_temperature_ts = theta * grad_temperature_star[q] + c1 * grad_temperature_n[q];

              const double inv_rho_ts = InlineFunctions::f(phi_ch_ts,
                                                           psi_ac_ts,
                                                           inv_density_l,
                                                           inv_density_s,
                                                           inv_density_g);
              const double rho_ts = 1./ inv_rho_ts;
              const double rho_ts_square = rho_ts * rho_ts;

              const double inv_rho_partial_phi_ch_ts =
                  InlineFunctions::f_partial_phi(phi_ch_ts,
                                                 psi_ac_ts,
                                                 inv_density_l,
                                                 inv_density_s,
                                                 inv_density_g);
              const double inv_rho_partial_psi_ac_ts =
                  InlineFunctions::f_partial_psi(phi_ch_ts,
                                                 psi_ac_ts,
                                                 inv_density_l,
                                                 inv_density_s);
              const double inv_rho_partial2_phi2_ts =
                  InlineFunctions::f_partial2_phi2(phi_ch_ts,
                                                   psi_ac_ts,
                                                   inv_density_l,
                                                   inv_density_s,
                                                   inv_density_g);
              const double inv_rho_partial2_psi2_ts =
                  InlineFunctions::f_partial2_psi2(phi_ch_ts,
                                                   psi_ac_ts,
                                                   inv_density_l,
                                                   inv_density_s);
              const double inv_rho_partial_phi_partial_psi =
                  InlineFunctions::f_partial_phi_partial_psi(phi_ch_ts,
                                                             psi_ac_ts,
                                                             inv_density_l,
                                                             inv_density_s);

              const double rho_partial_phi_ch_ts = -rho_ts_square * inv_rho_partial_phi_ch_ts;
              const double rho_partial_psi_ac_ts = -rho_ts_square * inv_rho_partial_psi_ac_ts;

              const double material_derivative_ts = ((phi_ch_star[q] - phi_ch_n[q])/present_timestep + vel_ts * grad_phi_ch_ts);
//              {
//                std::cout<<" phi_ch_star: "<<phi_ch_star[q]
//                           << " phi_ch_n: "<<phi_ch_n[q]
//                              << " present time step "<<present_timestep
//                              <<" vel_ts: "<< vel_ts
//                             <<" frad_phi_ch_ts: "<<grad_phi_ch_ts<<std::endl;
//              }

              const double h_ts   = rho_ts * material_derivative_ts;

              const Tensor<1,dim> grad_inv_rho_ts = inv_rho_partial_phi_ch_ts * grad_phi_ch_ts + inv_rho_partial_psi_ac_ts * grad_psi_ac_ts;

              const double c_ts = InlineFunctions::f(phi_ch_ts,
                                                     psi_ac_ts,
                                                     cl, cs, cg);
              const double c_partial_phi_ch_ts = InlineFunctions::f_partial_phi(phi_ch_ts,
                                                                             psi_ac_ts,
                                                                             cl, cs, cg);
              const double c_partial_psi_ac_ts = InlineFunctions::f_partial_psi(phi_ch_ts,
                                                                             psi_ac_ts,
                                                                             cl, cs);
              const double c_partial2_phi2_ts = InlineFunctions::f_partial2_phi2(phi_ch_ts,
                                                                              psi_ac_ts,
                                                                              cl, cs, cg);
              const double c_partial2_psi2_ts = InlineFunctions::f_partial2_psi2(phi_ch_ts,
                                                                              psi_ac_ts,
                                                                              cl, cs);
              const double c_partial_phi_partial_psi_ac_ts = InlineFunctions::f_partial_phi_partial_psi(phi_ch_ts,
                                                                                                  psi_ac_ts,
                                                                                                  cl, cs);

              const double eta_ts = InlineFunctions::g(phi_ch_ts,
                                                       psi_ac_ts,
                                                       eta_l,
                                                       eta_s,
                                                       eta_g);
              const double eta_partial_phi_ch_ts = InlineFunctions::g_pratial_phi(psi_ac_ts, eta_l, eta_s, eta_g);
              const double eta_partial_psi_ac_ts = InlineFunctions::g_pratial_psi(phi_ch_ts, eta_l, eta_s);

              // E_theta_star
              const Tensor<2,dim> e_ts = grad_vel_ts + transpose(grad_vel_ts) -  2./3. * div_vel_ts * id_tensor;

              const double r_phi_ch_ts = InlineFunctions::r(phi_ch_ts);
              const double r_psi_ac_ts = InlineFunctions::r(psi_ac_ts);
              const double r_prime_phi_ch_ts = InlineFunctions::r_prime(phi_ch_ts);
              const double r_prime_psi_ac_ts = InlineFunctions::r_prime(psi_ac_ts);
              const double r_prime_prime_phi_ch_ts = InlineFunctions::r_prime_prime(phi_ch_ts);
              const double r_prime_prime_psi_ac_ts = InlineFunctions::r_prime_prime(psi_ac_ts);
              const double w_psi_ac_ts = InlineFunctions::w(psi_ac_ts, eps);
              const double w_prime_psi_ac_ts = InlineFunctions::w_prime(psi_ac_ts, eps);
              const double w_prime_phi_ch_ts = InlineFunctions::w_prime(phi_ch_ts, eps);
              const double w_prime_prime_phi_ch_ts = InlineFunctions::w_prime_prime(phi_ch_ts, eps);
              const double w_prime_prime_psi_ac_ts = InlineFunctions::w_prime_prime(psi_ac_ts, eps);

              const Tensor<2,dim> gamma_ts = lambda_phi * outer_product(grad_phi_ch_ts, grad_phi_ch_ts)
                  + r_phi_ch_ts * lambda_psi * outer_product(grad_psi_ac_ts, grad_psi_ac_ts);

              const Tensor<1,dim> D_vel_Dt_ts = (vel_star[q] - vel_n[q])/present_timestep +  grad_vel_ts * vel_bar
                  + 0.5 * div_vel_bar * vel_ts;

              const double D_temperature_Dt_ts = (temperature_star[q] - temperature_n[q])/present_timestep + vel_ts * grad_temperature_ts;

              const double log_t_ts_tm = std::log(temperature_ts/melting_t);
              // TM(1-T_ts/TM) + T_ts*log(T_ts/TM), used in  w3 and w5
              const double w35_reuse_term1 = melting_t * (1. - temperature_ts/melting_t) + temperature_ts * log_t_ts_tm;
              // w(phi_ch_ts) + 0.5 |grad_psi_ac_ts|^2
              const double w3_reuse_term2 = w_psi_ac_ts + 0.5 * (grad_psi_ac_ts * grad_psi_ac_ts);

              // (psi_ac_ts - psi_ac_n)/dt + vel_ts * grad_psi_ac_ts
              const double D_psi_D_t_ts = (psi_ac_star[q] - psi_ac_n[q])/present_timestep + vel_ts * grad_psi_ac_ts;

              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  shape_phi_ch[k]       = fe_values[extractors.phi_ch].value(k, q);
                  shape_psi_ac[k]       = fe_values[extractors.psi_ac].value(k, q);
                  shape_phi_ch_theta[k] = theta * shape_phi_ch[k];
                  shape_psi_ac_theta[k] = theta * shape_psi_ac[k];

                  grad_shape_phi_ch[k]       = fe_values[extractors.phi_ch].gradient(k, q);
                  grad_shape_phi_ch_theta[k] = theta * grad_shape_phi_ch[k];
                  grad_shape_psi_ac[k]       = fe_values[extractors.psi_ac].gradient(k, q);
                  grad_shape_psi_ac_theta[k] = theta * grad_shape_psi_ac[k];

                  shape_mu_phi_ch[k]            = fe_values[extractors.mu_phi_ch].value(k, q);
                  shape_mu_psi_ac[k]            = fe_values[extractors.mu_psi_ac].value(k, q);

                  grad_shape_mu_phi_ch[k]       = fe_values[extractors.mu_phi_ch].gradient(k, q);
                  grad_shape_mu_psi_ac[k]       = fe_values[extractors.mu_psi_ac].gradient(k, q);

                  shape_vel[k]       = fe_values[extractors.velocities].value(k, q);
                  shape_vel_theta[k] = theta * shape_vel[k];
                  grad_shape_vel[k] = fe_values[extractors.velocities].gradient(k, q);
                  shape_div_vel[k]  = fe_values[extractors.velocities].divergence(k, q);
                  shape_pressure[k] = fe_values[extractors.pressure].value(k, q);

                  shape_temperature[k]             = fe_values[extractors.temperature].value(k, q);
                  shape_temperature_theta[k]       = theta * shape_temperature[k];
                  grad_shape_temperature[k]        = fe_values[extractors.temperature].gradient(k, q);
                  grad_shape_temperature_theta[k]  = theta * grad_shape_temperature[k];

                  shape_inv_rho_theta[k] = inv_rho_partial_phi_ch_ts * shape_phi_ch_theta[k]
                      + inv_rho_partial_psi_ac_ts * shape_psi_ac_theta[k];
                  shape_rho_theta[k]     = rho_partial_phi_ch_ts * shape_phi_ch_theta[k]
                      + rho_partial_psi_ac_ts * shape_psi_ac_theta[k];

                  shape_inv_rho_partial_phi_theta[k] = inv_rho_partial2_phi2_ts * shape_phi_ch_theta[k]
                      + inv_rho_partial_phi_partial_psi * shape_psi_ac_theta[k];
                  shape_inv_rho_partial_psi_theta[k] = inv_rho_partial_phi_partial_psi * shape_phi_ch_theta[k]
                      + inv_rho_partial2_psi2_ts * shape_psi_ac_theta[k];

                  shape_h_ts[k] = shape_rho_theta[k] * material_derivative_ts
                      + rho_ts * (shape_phi_ch[k]/present_timestep + shape_vel_theta[k] * grad_phi_ch_ts
                                  + vel_ts * grad_shape_phi_ch_theta[k]);
//                  {
//                    std::cout<<" shape_rho_theta: "<< shape_rho_theta[k]
//                               << " material_derivative_ts " << material_derivative_ts
//                               << " rho_ts " << rho_ts <<std::endl;

//                  }

                  grad_shape_inv_rho_theta[k] = shape_inv_rho_partial_phi_theta[k] * grad_phi_ch_ts
                      + shape_inv_rho_partial_psi_theta[k] * grad_psi_ac_ts
                      + inv_rho_partial_phi_ch_ts * grad_shape_phi_ch_theta[k]
                      + inv_rho_partial_psi_ac_ts * grad_shape_psi_ac_theta[k];

                  shape_c_theta[k] = c_partial_phi_ch_ts * shape_phi_ch_theta[k]
                      + c_partial_psi_ac_ts * shape_psi_ac_theta[k];
                  shape_c_partial_phi_theta[k] = c_partial2_phi2_ts * shape_phi_ch_theta[k]
                      + c_partial_phi_partial_psi_ac_ts * shape_psi_ac_theta[k];
                  shape_c_partial_psi_theta[k] = c_partial_phi_partial_psi_ac_ts * shape_phi_ch_theta[k]
                      + c_partial2_psi2_ts * shape_psi_ac_theta[k];

                  shape_eta_theta[k] = eta_partial_phi_ch_ts * shape_phi_ch_theta[k]
                      + eta_partial_psi_ac_ts * shape_psi_ac_theta[k];

                  shape_e_theta[k] = theta * (grad_shape_vel[k] + transpose(grad_shape_vel[k])
                                              - 2./3. * shape_div_vel[k] * id_tensor);

                  const Tensor<2,dim> grad_phi_ch_ts_grad_shape_phi_ch_theta = outer_product(grad_phi_ch_ts, grad_shape_phi_ch_theta[k]);
                  const Tensor<2,dim> grad_psi_ac_ts_grad_shape_psi_ac_theta = outer_product(grad_psi_ac_ts, grad_shape_psi_ac_theta[k]);
                  shape_gamma_theta[k] = lambda_phi * (grad_phi_ch_ts_grad_shape_phi_ch_theta
                                                       + transpose(grad_phi_ch_ts_grad_shape_phi_ch_theta))
                      + lambda_psi * (r_prime_phi_ch_ts * shape_phi_ch_theta[k] * outer_product(grad_psi_ac_ts, grad_psi_ac_ts)
                                      + r_phi_ch_ts * (grad_psi_ac_ts_grad_shape_psi_ac_theta + transpose(grad_psi_ac_ts_grad_shape_psi_ac_theta)));
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  if(assemble_matrix)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {

                          // w1,2
                          mat = shape_h_ts[j] * shape_phi_ch[i] + mobility_phi * grad_shape_mu_phi_ch[j] * grad_shape_phi_ch[i];
#if 1
                          // w3
                          //  term i
                          mat += shape_mu_phi_ch[j] * shape_mu_phi_ch[i];
                          //  term ii
                          mat += (-((r_prime_psi_ac_ts * r_prime_phi_ch_ts * shape_psi_ac_theta[j]
                                     + r_psi_ac_ts * r_prime_prime_phi_ch_ts * shape_phi_ch_theta[j])
                                    * latent_heat * (1. - temperature_ts/melting_t))
                                  + r_psi_ac_ts * r_prime_phi_ch_ts * latent_heat/melting_t * shape_temperature_theta[j]
                                  + (shape_c_partial_phi_theta[j] * w35_reuse_term1
                                     + c_partial_phi_ch_ts * shape_temperature_theta[j] * log_t_ts_tm)) * shape_mu_phi_ch[i];
                          //  term iii
                          mat += (-(lambda_phi * w_prime_prime_phi_ch_ts
                                    + lambda_psi * r_prime_prime_phi_ch_ts * w3_reuse_term2) * shape_phi_ch_theta[j]
                                  - lambda_psi * r_prime_phi_ch_ts * (w_prime_psi_ac_ts * shape_psi_ac_theta[j] + grad_psi_ac_ts * grad_shape_psi_ac_theta[j])
                                  - (shape_pressure[j] * inv_rho_partial_phi_ch_ts + pressure_star[q] * shape_inv_rho_partial_phi_theta[j])) * shape_mu_phi_ch[i];
                          //  term iv
                          mat += lambda_phi * (grad_shape_phi_ch_theta[j] * grad_shape_mu_phi_ch[i])
                              +  lambda_phi * shape_rho_theta[j] * (grad_phi_ch_ts * grad_inv_rho_ts) * shape_mu_phi_ch[i]
                              +  lambda_phi * rho_ts * (grad_shape_phi_ch_theta[j] * grad_inv_rho_ts) * shape_mu_phi_ch[i]
                              +  lambda_phi * rho_ts * (grad_phi_ch_ts * grad_shape_inv_rho_theta[j]) * shape_mu_phi_ch[i];

                          // todo: add face integral term v

                          // w4
                          mat += (rho_ts * (shape_psi_ac[j]/present_timestep + vel_ts * grad_shape_psi_ac_theta[j]
                                            + shape_vel_theta[j] * grad_psi_ac_ts)
                                  + shape_rho_theta[j] * D_psi_D_t_ts + mobility_psi * shape_mu_psi_ac[j]) * shape_psi_ac[i];

                          // w5
                          //  term i
                          mat += shape_mu_psi_ac[j] * shape_mu_psi_ac[i];
                          //  term ii
                          mat += ( -( (r_prime_prime_psi_ac_ts * r_phi_ch_ts * shape_psi_ac_theta[j]
                                       + r_prime_psi_ac_ts * r_prime_phi_ch_ts * shape_phi_ch_theta[j])
                                      * latent_heat * (1. - temperature_ts/melting_t))
                                   + r_prime_psi_ac_ts * r_phi_ch_ts * latent_heat/melting_t * shape_temperature_theta[j]
                                   + (shape_c_partial_psi_theta[j] * w35_reuse_term1
                                      + c_partial_psi_ac_ts * shape_temperature_theta[j] * log_t_ts_tm)) * shape_mu_psi_ac[i];
                          //  term iii
                          mat += (-(r_prime_phi_ch_ts * shape_phi_ch_theta[j] * lambda_psi * w_prime_psi_ac_ts
                                    + r_phi_ch_ts * lambda_psi * w_prime_prime_psi_ac_ts * shape_psi_ac_theta[j])
                                  -  (shape_pressure[j] * inv_rho_partial_psi_ac_ts + pressure_star[q] * shape_inv_rho_partial_psi_theta[j])) * shape_mu_psi_ac[i];
                          //  term iv
                          mat += - (r_prime_phi_ch_ts * shape_phi_ch_theta[j] * lambda_psi * grad_psi_ac_ts
                                    + r_phi_ch_ts * lambda_psi * grad_shape_psi_ac_theta[j]) * grad_shape_mu_psi_ac[i]
                              - lambda_psi * shape_rho_theta[j] * r_phi_ch_ts * (grad_psi_ac_ts * grad_inv_rho_ts) * shape_mu_psi_ac[i]
                              - lambda_psi * rho_ts * r_prime_phi_ch_ts * shape_phi_ch_theta[j] * (grad_psi_ac_ts * grad_inv_rho_ts) * shape_mu_psi_ac[i]
                              - lambda_psi * rho_ts * r_phi_ch_ts * (grad_shape_psi_ac_theta[j] * grad_inv_rho_ts) * shape_mu_psi_ac[i]
                              - lambda_psi * rho_ts * r_phi_ch_ts * (grad_psi_ac_ts * grad_shape_inv_rho_theta[j]) * shape_mu_psi_ac[i];

                          // w6
                          mat += (shape_rho_theta[j] * D_vel_Dt_ts
                                  + rho_ts * (shape_vel[j]/present_timestep + grad_shape_vel[j] * vel_bar * theta
                                              + 0.5 * div_vel_bar * shape_vel_theta[j])) * shape_vel[i]
                              -  shape_pressure[j] * shape_div_vel[i]
                              +  scalar_product(shape_eta_theta[j] * e_ts + eta_ts * shape_e_theta[j], grad_shape_vel[i])
                              -  scalar_product(shape_rho_theta[j] * gamma_ts + rho_ts * shape_gamma_theta[j], grad_shape_vel[i]);

                          // w7
                          mat += (- theta * shape_div_vel[j]
                                  + (shape_inv_rho_partial_phi_theta[j] * h_ts + inv_rho_partial_phi_ch_ts * shape_h_ts[j])
                                  - (shape_inv_rho_partial_psi_theta[j] * mobility_psi * mu_psi_ac_star[q]
                                     + inv_rho_partial_psi_ac_ts * mobility_psi * shape_mu_psi_ac[j])) * shape_pressure[i];

                          // w8
                          //  term i
                          mat += ((shape_rho_theta[j] * c_ts + rho_ts * shape_c_theta[j]) * D_temperature_Dt_ts
                                  +  rho_ts * c_ts * (shape_temperature[j]/present_timestep + shape_vel_theta[j] * grad_temperature_ts
                                                      + vel_ts * grad_shape_temperature_theta[j])) * shape_temperature[i];
                          //  term ii
                          mat +=(-2. * ( mobility_phi * (grad_mu_phi_ch_star[q] * grad_shape_mu_phi_ch[j])
                                         + mobility_psi * mu_psi_ac_star[q] * shape_mu_psi_ac[j])
                                 -  shape_eta_theta[j] * scalar_product(e_ts, grad_vel_ts)
                                 -  eta_ts * scalar_product(shape_e_theta[j], grad_vel_ts)
                                 - eta_ts * scalar_product(e_ts, grad_shape_vel[j]) * theta) * shape_temperature[i];
                          //  term iii
                          mat += thermal_conductivity * grad_shape_temperature_theta[j] * grad_shape_temperature[i];
                          //  term iv
                          mat += latent_heat/melting_t * (r_prime_psi_ac_ts * shape_psi_ac_theta[j] * r_prime_phi_ch_ts * h_ts
                                                          + r_psi_ac_ts * r_prime_prime_phi_ch_ts * shape_phi_ch_theta[j] * h_ts
                                                          + r_psi_ac_ts * r_prime_phi_ch_ts * shape_h_ts[j]
                                                          - r_prime_prime_psi_ac_ts * shape_psi_ac_theta[j] * r_phi_ch_ts * mobility_psi * mu_psi_ac_star[q]
                                                          - r_prime_psi_ac_ts * r_prime_phi_ch_ts * shape_phi_ch_theta[j] * mobility_psi * mu_psi_ac_star[q]
                                                          - r_prime_psi_ac_ts * r_phi_ch_ts * mobility_psi * shape_mu_psi_ac[j])
                              * temperature_ts * shape_temperature[i]
                              + latent_heat/melting_t * (r_psi_ac_ts * r_prime_phi_ch_ts * h_ts
                                                         - r_prime_psi_ac_ts * r_phi_ch_ts * mobility_psi * mu_psi_ac_star[q])
                              * shape_temperature_theta[j] * shape_temperature[i];
                          //  term v
                          mat += (shape_c_partial_phi_theta[j] * h_ts + c_partial_phi_ch_ts * shape_h_ts[j]
                                  - shape_c_partial_psi_theta[j] * mobility_psi * mu_psi_ac_star[q]
                                  - c_partial_psi_ac_ts * mobility_psi * shape_mu_psi_ac[j])
                              *  temperature_ts * log_t_ts_tm * shape_temperature[i]
                              +  (c_partial_phi_ch_ts * h_ts - c_partial_psi_ac_ts * mobility_psi * mu_psi_ac_star[q])
                              *  (log_t_ts_tm + 1.) * shape_temperature_theta[j] * shape_temperature[i];
#endif
                          cell_matrix(i,j) += mat * fe_values.JxW(q);

                        }
                    }

                  // w1,2
                  rhs = - h_ts * shape_phi_ch[i] - mobility_phi * (grad_mu_phi_ch_star[q] * grad_shape_phi_ch[i]);

                  // w3
                  //  term i
                  rhs += - mu_phi_ch_star[q] * shape_mu_phi_ch[i];
                  //  term ii
                  rhs += (r_psi_ac_ts * r_prime_phi_ch_ts * latent_heat * (1. - temperature_ts/melting_t)
                          - c_partial_phi_ch_ts * w35_reuse_term1) * shape_mu_phi_ch[i];
                  //  term iii
                  rhs += (lambda_phi * w_prime_phi_ch_ts + lambda_psi * r_prime_phi_ch_ts * w3_reuse_term2
                          + pressure_star[q] * inv_rho_partial_phi_ch_ts) * shape_mu_phi_ch[i];
                  //  term iv
                  rhs += lambda_phi * (grad_phi_ch_ts * grad_shape_mu_phi_ch[i])
                      +  lambda_phi * rho_ts * (grad_phi_ch_ts * grad_inv_rho_ts) * shape_mu_phi_ch[i];

                  // w4
                  rhs +=  (- rho_ts *  D_psi_D_t_ts - mobility_psi * mu_psi_ac_star[q]) * shape_mu_psi_ac[i];

                  // w5
                  //  term i
                  rhs += - mu_psi_ac_star[q] * shape_mu_psi_ac[i];
                  //  term ii
                  rhs += (r_prime_psi_ac_ts * r_phi_ch_ts * latent_heat * (1. - temperature_ts/melting_t)
                          - c_partial_psi_ac_ts * w35_reuse_term1) * shape_mu_psi_ac[i];
                  //  term iii
                  rhs += (r_phi_ch_ts * lambda_psi * w_prime_psi_ac_ts + pressure_star[q] * inv_rho_partial_psi_ac_ts) * shape_mu_psi_ac[i];
                  //  term iv
                  rhs += lambda_psi * r_phi_ch_ts * (grad_psi_ac_ts * grad_shape_mu_psi_ac[i])
                      +  lambda_psi * rho_ts * r_phi_ch_ts * (grad_psi_ac_ts * grad_inv_rho_ts) * shape_mu_psi_ac[i];

                  // w6
                  rhs += - rho_ts * D_vel_Dt_ts * shape_vel[i]
                      +    pressure_star[q] * shape_div_vel[i]
                      -    eta_ts * scalar_product(e_ts, grad_shape_vel[i])
                      +    rho_ts * scalar_product(gamma_ts, grad_shape_vel[i]);

                  // w7
                  rhs += (div_vel_ts - inv_rho_partial_phi_ch_ts * h_ts
                          +  inv_rho_partial_psi_ac_ts * mobility_psi * mu_psi_ac_star[q]) * shape_pressure[i];

                  // w8
                  //  term i
                  rhs += - rho_ts * c_ts * D_temperature_Dt_ts * shape_temperature[i];
                  //  term ii
                  rhs += (mobility_phi * (grad_mu_phi_ch_star[q] * grad_mu_phi_ch_star[q])
                          + mobility_psi * (mu_psi_ac_star[q] * mu_psi_ac_star[q])
                          + eta_ts * scalar_product(e_ts, grad_vel_ts)) * shape_temperature[i];
                  //  term iii
                  rhs += - thermal_conductivity * grad_temperature_ts * grad_shape_temperature[i];
                  //  term iv
                  rhs += - (latent_heat * r_psi_ac_ts * r_prime_phi_ch_ts * h_ts
                            - r_prime_psi_ac_ts * r_phi_ch_ts * mobility_psi * mu_psi_ac_star[q])
                      * temperature_ts/melting_t * shape_temperature[i];
                  // term v
                  rhs += - (c_partial_phi_ch_ts * h_ts - c_partial_psi_ac_ts * mobility_psi * mu_psi_ac_star[q])
                      * temperature_ts * log_t_ts_tm * shape_temperature[i];


                  cell_rhs(i) += rhs * fe_values.JxW(q);
                }
              }

            constraints_newton_update.distribute_local_to_global(cell_matrix,
                                                                 cell_rhs,
                                                                 local_dof_indices,
                                                                 system_matrix,
                                                                 system_rhs);
          }// cell loop

    system_matrix.compress(VectorOperation::add);
//    preconditioner_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}

template <int dim>
void StokesProblem<dim>::newton_iteration()
{
  TimerOutput::Scope t(computing_timer, "Newton");

  // set to 1 for testing
  const unsigned int max_iter = 10;
  bool assemble_matrix = false;
  assemble_system(assemble_matrix);
  double residual = system_rhs.l2_norm();

  pcout << "initial residual=" << residual << std::endl;

#ifdef USE_PETSC_LA
  SolverControl cn;
  PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
#else
  TrilinosWrappers::SolverDirect::AdditionalData data;
  data.solver_type = "Amesos_Lapack";
  SolverControl                  solver_control(1000, 1e-10);
  TrilinosWrappers::SolverDirect solver(solver_control, data);
#endif

  assemble_matrix = true;
  VectorType locally_owned_solution(system_rhs);
  locally_owned_solution = current_solution;
  for (unsigned int k = 1; k <= max_iter; ++k) {
      if (residual < 1.e-6 * hmin)
        break;
      assemble_system(assemble_matrix);
      solver.solve(system_matrix, newton_update, system_rhs);

      pcout<<" mat norm: "<<system_matrix.frobenius_norm()
          <<" rh2 norm: "<< system_rhs.l2_norm()
            <<" sol norm: "<< newton_update.l2_norm()<<std::endl;

      constraints_newton_update.distribute(newton_update);

      locally_owned_solution += newton_update;
      current_solution = locally_owned_solution;

      residual = system_rhs.l2_norm();
      pcout << "k= " << k << "  residual = " << residual <<  std::endl;

    }

}

template <int dim>
void
StokesProblem<dim>::map_dofs_to_component(
  const DoFHandler<dim> &    dof,
  std::vector<unsigned int> &global_index_to_component)
{
  std::vector<types::global_dof_index> local_dof_indices;
  // store the components of each global index.
  for (const auto &cell : dof.active_cell_iterators())
    {
      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < cell->get_fe().dofs_per_cell; ++i)
        {
          const unsigned int dof_comp =
            cell->get_fe().system_to_component_index(i).first;
          global_index_to_component[local_dof_indices[i]] = dof_comp;
        }
    }
}

template <int dim>
void StokesProblem<dim>::solve()
{
//    TimerOutput::Scope t(computing_timer, "solve");

//    LA::MPI::PreconditionAMG prec_A;
//    {
//        LA::MPI::PreconditionAMG::AdditionalData data;

//#ifdef USE_PETSC_LA
//        data.symmetric_operator = true;
//#endif
//        prec_A.initialize(system_matrix.block(0, 0), data);
//    }

//    LA::MPI::PreconditionAMG prec_S;
//    {
//        LA::MPI::PreconditionAMG::AdditionalData data;

//#ifdef USE_PETSC_LA
//        data.symmetric_operator = true;
//#endif
//        prec_S.initialize(preconditioner_matrix.block(1, 1), data);
//    }

//    using mp_inverse_t = LinearSolvers::InverseMatrix<LA::MPI::SparseMatrix,
//          LA::MPI::PreconditionAMG>;
//    const mp_inverse_t mp_inverse(preconditioner_matrix.block(1, 1), prec_S);

//    const LinearSolvers::BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG,
//          mp_inverse_t>
//          preconditioner(prec_A, mp_inverse);

//    SolverControl solver_control(system_matrix.m(),
//                                 1e-10 * system_rhs.l2_norm());

//    SolverMinRes<LA::MPI::BlockVector> solver(solver_control);

//    LA::MPI::BlockVector distributed_solution(block_owned_partitioning,
//            mpi_communicator);

//    constraints.set_zero(distributed_solution);

//    solver.solve(system_matrix,
//                 distributed_solution,
//                 system_rhs,
//                 preconditioner);

//    pcout << "   Solved in " << solver_control.last_step() << " iterations."
//          << std::endl;

//    constraints.distribute(distributed_solution);

//    locally_relevant_solution = distributed_solution;
//    const double mean_pressure =
//        VectorTools::compute_mean_value(dof_handler,
//                                        QGauss<dim>(velocity_degree + 2),
//                                        locally_relevant_solution,
//                                        dim);
//    distributed_solution.block(1).add(-mean_pressure);
//    locally_relevant_solution.block(1) = distributed_solution.block(1);
}



template <int dim>
void StokesProblem<dim>::refine_grid()
{
    TimerOutput::Scope t(computing_timer, "refine");

    triangulation.refine_global();
}



template <int dim>
void StokesProblem<dim>::output_results(const unsigned int cycle) const
{
    AssertDimension(extractors.velocities.first_vector_component, 0);
    std::vector<std::string> solution_names(dim, "velocity");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);


    for(unsigned int n=dim; n<fe.n_components(); ++n)
      {
        if(n == extractors.pressure.component)
          solution_names.emplace_back("pressure");
        else if(n == extractors.temperature.component)
          solution_names.emplace_back("temperature");
        else if(n == extractors.phi_ch.component)
          solution_names.emplace_back("phi_ch");
        else if(n == extractors.mu_phi_ch.component)
          solution_names.emplace_back("mu_phi_ch");
        else if(n == extractors.psi_ac.component)
          solution_names.emplace_back("psi_ac");
        else if(n == extractors.mu_psi_ac.component)
          solution_names.emplace_back("mu_psi_ac");
        else
          {
            ExcNotImplemented("output name no match!");
          }
        data_component_interpretation.push_back(
            DataComponentInterpretation::component_is_scalar);

      }

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
        subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
        "./", "solution", cycle, mpi_communicator, 2);
}



template <int dim>
void StokesProblem<dim>::run()
{
#ifdef USE_PETSC_LA
    pcout << "Running using PETSc." << std::endl;
#else
    pcout << "Running using Trilinos." << std::endl;
#endif
    pcout << "n refinement " << n_refinement << ':' << std::endl;

    unsigned int step_number = 0;
    double runtime           = 0.;
    present_timestep = 0.001;

    const unsigned int max_step_number = 10;
    const unsigned int output_interval = 1;

    make_grid();

#ifdef USE_DIRECT_SOLVER
    setup_system();
#else
    setup_block_system();
#endif

    setup_initial_condition();

    output_results(step_number);

    while (step_number < max_step_number)
      {
        old_timestep     = present_timestep;

        step_number ++;
        runtime += present_timestep;
        pcout << "###starting step " << step_number << "  dt=" << present_timestep
              << "  time=" << runtime << std::endl;

        old_old_solution = old_solution;          // n-1
        old_solution = locally_relevant_solution; // n
        current_solution = old_solution; // u^*, newton initial guess

        newton_iteration();

        if (step_number % output_interval == 0)
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(step_number);
          }

      }
    computing_timer.print_summary();
    computing_timer.reset();

    pcout << std::endl;
}
} // namespace Step55



int main(int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace Step55;

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        const TestCase testcase = TestCase::test1;
        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "running " << enum_str[static_cast<int>(testcase)]
                    << std::endl;
        StokesProblem<2> problem(2, testcase);
        problem.run();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}

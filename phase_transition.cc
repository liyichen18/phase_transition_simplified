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

/* #define FORCE_USE_OF_TRILINOS */

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
using namespace dealii::LinearAlgebraPETSc;
//#  define USE_PETSC_LA
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
    if(phi<0.)
      return 0.;
    else if(phi>1.)
      return 1.;
    else
      return 30. * phi * phi * (1. - phi) * (1. - phi);
  }

  inline
  double r_prime_prime(const double phi)
  {
    if(phi<0.)
      return 0.;
    else if(phi>1.)
      return 1.;
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
    return  r_prime(phi) * (r(psi) * (fl - fs) - fg);
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
    return  r_prime_prime(phi) * (r(psi) * (fl - fs) - fg);
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
    return  psi * (gl - gg) + gs;
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
      InitialValues (const double epsilon, const TestCase testcase, const Extractors<dim> &ex)
                  : Function<dim>(dim + 6),
                    eps(epsilon),
                    test_case(testcase),
                    extractors(ex)
                    {}

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &  value) const override;
  private:
      const double eps;
      const TestCase test_case;
      const Extractors<dim> extractors;

  };

  template <int dim>
  void InitialValues<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &  values) const
  {
      const double eps1=eps*sqrt(2);  //sqrt(2)* eps
      double d=0;

      switch (test_case)
        {
        case TestCase::test1: {
            Point<dim> center, axes;
            double radius;
            switch (dim)
              {
              case 2: //2D case
                {
                  center=Point<dim>(0,0);
                  axes=Point<dim>(0.5,0.5);
                  radius=sqrt(axes(0)*axes(1));
                  break;
                }
              case 3: //3D case
                {
                  center=Point<dim>(0,0,0);
                  axes=Point<dim>(0.5,0.5,0.5);
                  radius=pow(axes(0)*axes(1)*axes(2),1./3.);
                  break;
                }
              }
            for(unsigned int i=0; i<dim; i++)
              d+=pow(p(i)/axes(i),2);
            d=(sqrt(d)-1)*radius;

            break;
          }
        case TestCase::test2: {

            break;
          }
        default:
          Assert(false, ExcNotImplemented("Please choose the right test case"));
        }
      double phi=-tanh(d/eps1);
      values(extractors.phi_ch.component) = phi;

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
class RightHandSide : public Function<dim>
{
public:
    RightHandSide()
        : Function<dim>(dim + 1)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
};


template <int dim>
void RightHandSide<dim>::vector_value(const Point<dim> &p,
                                      Vector<double> &  values) const
{
    const double R_x = p[0];
    const double R_y = p[1];

    const double pi  = numbers::PI;
    const double pi2 = pi * pi;
    values[0] =
        -1.0L / 2.0L * (-2 * sqrt(25.0 + 4 * pi2) + 10.0) *
        exp(R_x * (-2 * sqrt(25.0 + 4 * pi2) + 10.0)) -
        0.4 * pi2 * exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi) +
        0.1 * pow(-sqrt(25.0 + 4 * pi2) + 5.0, 2) *
        exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi);
    values[1] = 0.2 * pi * (-sqrt(25.0 + 4 * pi2) + 5.0) *
                exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) -
                0.05 * pow(-sqrt(25.0 + 4 * pi2) + 5.0, 3) *
                exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) /
                pi;
    values[2] = 0;
}


template <int dim>
class ExactSolution : public Function<dim>
{
public:
    ExactSolution()
        : Function<dim>(dim + 1)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
};

template <int dim>
void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                      Vector<double> &  values) const
{
    const double R_x = p[0];
    const double R_y = p[1];

    const double pi  = numbers::PI;
    const double pi2 = pi * pi;
    values[0] =
        -exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * cos(2 * R_y * pi) + 1;
    values[1] = (1.0L / 2.0L) * (-sqrt(25.0 + 4 * pi2) + 5.0) *
                exp(R_x * (-sqrt(25.0 + 4 * pi2) + 5.0)) * sin(2 * R_y * pi) /
                pi;
    values[2] =
        -1.0L / 2.0L * exp(R_x * (-2 * sqrt(25.0 + 4 * pi2) + 10.0)) -
        2.0 *
        (-6538034.74494422 +
         0.0134758939981709 * exp(4 * sqrt(25.0 + 4 * pi2))) /
        (-80.0 * exp(3 * sqrt(25.0 + 4 * pi2)) +
         16.0 * sqrt(25.0 + 4 * pi2) * exp(3 * sqrt(25.0 + 4 * pi2))) -
        1634508.68623606 * exp(-3.0 * sqrt(25.0 + 4 * pi2)) /
        (-10.0 + 2.0 * sqrt(25.0 + 4 * pi2)) +
        (-0.00673794699908547 * exp(sqrt(25.0 + 4 * pi2)) +
         3269017.37247211 * exp(-3 * sqrt(25.0 + 4 * pi2))) /
        (-8 * sqrt(25.0 + 4 * pi2) + 40.0) +
        0.00336897349954273 * exp(1.0 * sqrt(25.0 + 4 * pi2)) /
        (-10.0 + 2.0 * sqrt(25.0 + 4 * pi2));
}



template <int dim>
class StokesProblem
{
public:
    StokesProblem(unsigned int velocity_degree, const TestCase & testcase);

    void run();

private:
    void make_grid();
    void setup_system();
    void setup_initial_condition();
    void make_boundary_constraints();
    Table<2, DoFTools::Coupling> make_coupling();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    std::vector<const FiniteElement<dim> *>
    create_fe_list(const unsigned int velocity_degree);

    std::vector<unsigned int> create_fe_multiplicities();



    unsigned int velocity_degree;
    double       viscosity;
    MPI_Comm     mpi_communicator;

    FESystem<dim>                             fe;
    parallel::distributed::Triangulation<dim> triangulation;
    DoFHandler<dim>                           dof_handler;

    std::vector<IndexSet> owned_partitioning;
    std::vector<IndexSet> relevant_partitioning;

    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LA::MPI::BlockSparseMatrix system_matrix;
    LA::MPI::BlockSparseMatrix preconditioner_matrix;
    LA::MPI::BlockVector       locally_relevant_solution; //u_n+1
    LA::MPI::BlockVector       old_solution; //u_n
    LA::MPI::BlockVector       current_solution; //u_*
    LA::MPI::BlockVector       system_rhs;
    LA::MPI::BlockVector       newton_update;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    const ComponentIndices<dim> component_ids;
    const Extractors<dim> extractors;

    const double eps;

    const TestCase test_case;

    const unsigned n_refinement;

    const double theta = 1.;

    const double density_s, density_l, density_g;
    const double inv_density_s, inv_density_l, inv_density_g;
    const double cl,cs,cg;
    const double eta_l,eta_s,eta_g;

    double present_timestep;
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
    , eps(0.01)
    , test_case(testcase)
    , n_refinement(7)
    , density_s(2.)
    , density_l(1.)
    , density_g(0.1)
    , inv_density_s(1./density_s)
    , inv_density_l(1./density_l)
    , inv_density_g(1./density_g)
    , present_timestep(0.001)
    , cl(1.), cs(1.), cg(1.)
    , eta_l(1.), eta_s(1.), eta_g(1.)
{




}


template <int dim>
void StokesProblem<dim>::make_grid()
{
    GridGenerator::hyper_cube(triangulation, 0., 1.);
    triangulation.refine_global(7);
}

template <int dim>
void StokesProblem<dim>::setup_initial_condition()
{
  LA::MPI::BlockVector tmp_initial_sol;
  tmp_initial_sol.reinit(owned_partitioning, mpi_communicator);
  VectorTools::interpolate(dof_handler,
                           InitialConditions::InitialValues<dim>(eps, test_case, extractors),
                           tmp_initial_sol);

  locally_relevant_solution = tmp_initial_sol;
}

template <int dim>
void StokesProblem<dim>::make_boundary_constraints()
{
  //todo: problem dependent
  constraints.reinit(locally_relevant_dofs);
  FEValuesExtractors::Vector vel(extractors.velocities);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           InitialConditions::InitialValues<dim>(eps, test_case, extractors),
                                           constraints,
                                           fe.component_mask(extractors.velocities));
  constraints.close();


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
                   || d == component_ids.temperature))
        coupling[c][d] = DoFTools::always;
      else if (c == component_ids.mu_psi_ac
               && (d == component_ids.mu_psi_ac
                   || d == component_ids.phi_ch
                   || d == component_ids.psi_ac
                   || d == component_ids.temperature))
        coupling[c][d] = DoFTools::always;
      else if (c == component_ids.phi_ch
               && (d == component_ids.phi_ch
                   || d == component_ids.mu_phi_ch))
        coupling[c][d] = DoFTools::always;
      else if (c == component_ids.psi_ac
               && (d == component_ids.psi_ac
                   || d == component_ids.mu_psi_ac))
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
void StokesProblem<dim>::setup_system()
{
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

    owned_partitioning.resize(n_blocks);

    unsigned int block_starting_id = 0;
    for(unsigned int i=0; i<n_blocks; ++i)
      {
        owned_partitioning[i] =
            dof_handler.locally_owned_dofs().get_view(block_starting_id,
                                                      block_starting_id + dofs_per_block[i]);
        block_starting_id += dofs_per_block[i];
        pcout<<" owned partitioning size ( "<<i<<" ): "<<owned_partitioning[i].size()<<std::endl;
      }

    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    relevant_partitioning.resize(n_blocks);

    block_starting_id = 0;
    for(unsigned int i=0; i<n_blocks; ++i)
      {
        relevant_partitioning[i] = locally_relevant_dofs.get_view(block_starting_id,
                                                                  block_starting_id + dofs_per_block[i]);
        block_starting_id += dofs_per_block[i];
//        pcout<<" relevant partitioning size ( "<<i<<" ): "<<relevant_partitioning[i].size()<<std::endl;
      }

    make_boundary_constraints();


    {
        system_matrix.clear();

        //todo: make coupling
        const Table<2, DoFTools::Coupling> coupling = make_coupling();

        BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

        DoFTools::make_sparsity_pattern(
            dof_handler, coupling, dsp, constraints, false);

        SparsityTools::distribute_sparsity_pattern(
            dsp,
            dof_handler.locally_owned_dofs(),
            mpi_communicator,
            locally_relevant_dofs);

//        std::cout<<" rows size: "<<owned_partitioning.size()
//                << " n block rows: "<<dsp.n_block_rows()<<std::endl;
        system_matrix.reinit(owned_partitioning, dsp, mpi_communicator);
    }

    locally_relevant_solution.reinit(owned_partitioning,
                                     relevant_partitioning,
                                     mpi_communicator);
    old_solution.reinit(owned_partitioning,
                        relevant_partitioning,
                        mpi_communicator);
    current_solution.reinit(owned_partitioning,
                        relevant_partitioning,
                        mpi_communicator);
    system_rhs.reinit(owned_partitioning, mpi_communicator);
    newton_update.reinit(owned_partitioning, mpi_communicator);
}



template <int dim>
void StokesProblem<dim>::assemble_system()
{
    TimerOutput::Scope t(computing_timer, "assembly");

    system_matrix         = 0;
    preconditioner_matrix = 0;
    system_rhs            = 0;

    SymmetricTensor<2, dim> id_tensor;
    for (unsigned int d = 0; d < dim; ++d)
      id_tensor[d][d] = 1.;

    // todo: check if it's accurate enough
    const QGauss<dim> quadrature_formula(velocity_degree + 2);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix2(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<double>         phi_star(n_q_points);
    std::vector<double>         phi_n(n_q_points);
    std::vector<double>         psi_star(n_q_points);
    std::vector<double>         psi_n(n_q_points);
    std::vector<Tensor<1,dim>>  grad_phi_star(n_q_points);
    std::vector<Tensor<1,dim>>  grad_phi_n(n_q_points);
    std::vector<Tensor<1,dim>>  grad_psi_star(n_q_points);
    std::vector<Tensor<1,dim>>  grad_psi_n(n_q_points);

    std::vector<Tensor<1,dim>>  vel_star(n_q_points);
    std::vector<Tensor<1,dim>>  vel_n(n_q_points);
    std::vector<Tensor<2,dim>>  grad_vel_star(n_q_points);
    std::vector<Tensor<2,dim>>  grad_vel_n(n_q_points);

    // shape functions:
    std::vector<double> shape_phi(dofs_per_cell);
    std::vector<double> shape_psi(dofs_per_cell);
    std::vector<double> shape_phi_theta(dofs_per_cell);
    std::vector<double> shape_psi_theta(dofs_per_cell);

    std::vector<Tensor<1,dim>> grad_shape_phi(dofs_per_cell);
    std::vector<Tensor<1,dim>> grad_shape_phi_theta(dofs_per_cell);
    std::vector<Tensor<1,dim>> grad_shape_psi(dofs_per_cell);
    std::vector<Tensor<1,dim>> grad_shape_psi_theta(dofs_per_cell);

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

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
            cell_matrix  = 0;
            cell_matrix2 = 0;
            cell_rhs     = 0;

            fe_values.reinit(cell);
            fe_values[extractors.phi_ch].get_function_values(current_solution, phi_star);
            fe_values[extractors.phi_ch].get_function_values(old_solution, phi_n);
            fe_values[extractors.psi_ac].get_function_values(current_solution, psi_star);
            fe_values[extractors.psi_ac].get_function_values(old_solution, psi_n);

            fe_values[extractors.phi_ch].get_function_gradients(current_solution, grad_phi_star);
            fe_values[extractors.phi_ch].get_function_gradients(old_solution, grad_phi_n);
            fe_values[extractors.psi_ch].get_function_gradients(current_solution, grad_psi_star);
            fe_values[extractors.psi_ch].get_function_gradients(old_solution, grad_psi_n);

            fe_values[extractors.velocities].get_function_values(current_solution, vel_star);
            fe_values[extractors.velocities].get_function_values(old_solution, vel_n);
            fe_values[extractors.velocities].get_function_gradients(current_solution, grad_vel_star);
            fe_values[extractors.velocities].get_function_gradients(old_solution, grad_vel_n);

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
              // ts: theta star
              const double c1 = 1.-theta;
              const double phi_ts = theta * phi_star[q] + c1 * phi_n[q];
              const double psi_ts = theta * psi_star[q] + c1 * psi_n[q];
              const Tensor<1,dim> grad_phi_ts = theta * grad_phi_star[q] + c1 * grad_phi_n[q];
              const Tensor<1,dim> grad_psi_ts = theta * grad_psi_star[q] + c1 * grad_psi_n[q];
              const Tensor<1,dim> vel_ts = theta * vel_star[q] + c1 * vel_n[q];
              const Tensor<2,dim> grad_vel_ts = theta * grad_vel_star[q] + c1 * grad_vel_n[q];
              const double div_vel_ts = trace(grad_vel_ts);

              const double inv_rho_ts = InlineFunctions::f(phi_ts,
                                                                   psi_ts,
                                                                   inv_density_l,
                                                                   inv_density_s,
                                                                   inv_density_g);
              const double rho_ts = 1./ inv_rho_ts;
              const double rho_ts_square = rho_ts * rho_ts;

              const double inv_rho_partial_phi_ts =
                  InlineFunctions::f_partial_phi(phi_ts,
                                                 psi_ts,
                                                 inv_density_l,
                                                 inv_density_s,
                                                 inv_density_g);
              const double inv_rho_partial_psi_ts =
                  InlineFunctions::f_partial_psi(phi_ts,
                                                 psi_ts,
                                                 inv_density_l,
                                                 inv_density_s);
              const double inv_rho_partial2_phi2_ts =
                  InlineFunctions::f_partial2_phi2(phi_ts,
                                                   psi_ts,
                                                   inv_density_l,
                                                   inv_density_s,
                                                   inv_density_g);
              const double inv_rho_partial2_psi2_ts =
                  InlineFunctions::f_partial2_psi2(phi_ts,
                                                   psi_ts,
                                                   inv_density_l,
                                                   inv_density_s);
              const double inv_rho_partial_phi_partial_psi =
                  InlineFunctions::f_partial_phi_partial_psi(phi_ts,
                                                             psi_ts,
                                                             inv_density_l,
                                                             inv_density_l);

              const double rho_partial_phi_ts = -rho_ts_square * inv_rho_partial_phi_ts;
              const double rho_partial_psi_ts = -rho_ts_square * inv_rho_partial_psi_ts;

              const double material_derivative_ts = ((phi_star[q] - phi_n[q])/present_timestep + vel_ts * grad_phi_ts);
              const double h_ts = rho_ts * material_derivative_ts;

              const Tensor<1,dim> grad_inv_rho_ts = inv_rho_partial_phi_ts * grad_phi_ts + inv_rho_partial_psi_ts * grad_psi_ts;

              const double c_ts = InlineFunctions::f(phi_ts,
                                                     psi_ts,
                                                     cl, cs, cg);
              const double c_partial_phi_ts = InlineFunctions::f_partial_phi(phi_ts,
                                                                             psi_ts,
                                                                             cl, cs, cg);
              const double c_partial_psi_ts = InlineFunctions::f_partial_psi(phi_ts,
                                                                             psi_ts,
                                                                             cl, cs);
              const double c_partial2_phi2_ts = InlineFunctions::f_partial2_phi2(phi_ts,
                                                                              psi_ts,
                                                                              cl, cs, cg);
              const double c_partial2_psi2_ts = InlineFunctions::f_partial2_psi2(phi_ts,
                                                                              psi_ts,
                                                                              cl, cs);
              const double c_partial_phi_partial_psi_ts = InlineFunctions::f_partial_phi_partial_psi(phi_ts,
                                                                                                  psi_ts,
                                                                                                  cl, cs);

              const double eta_ts = InlineFunctions::g(phi_ts,
                                                       psi_ts,
                                                       eta_l,
                                                       eta_s,
                                                       eta_g);
              const double eta_partial_phi_ts = InlineFunctions::g_pratial_phi(psi_ts, eta_l, eta_s, eta_g);
              const double eta_partial_psi_ts = InlineFunctions::g_pratial_psi(phi_ts, eta_l, eta_s);

              // E_theta_star
              const Tensor<2,dim> e_ts = grad_vel_ts + transpose(grad_vel_ts) -  2./3. * div_vel_ts * id_tensor;

              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  shape_phi[k]       = fe_values[extractors.phi_ch].value(k, q);
                  shape_psi[k]       = fe_values[extractors.psi_ch].value(k, q);
                  shape_phi_theta[k] = theta * shape_phi[k];
                  shape_psi_theta[k] = theta * shape_psi[k];

                  grad_shape_phi[k]       = fe_values[extractors.phi_ch].gradient(k, q);
                  grad_shape_phi_theta[k] = theta * grad_shape_phi[k];
                  grad_shape_psi[k]       = fe_values[extractors.psi_ch].gradient(k, q);
                  grad_shape_psi_theta[k] = theta * grad_shape_psi[k];

                  shape_vel[k]       = fe_values[extractors.velocities].value(k, q);
                  shape_vel_theta[k] = theta * shape_vel[k];
                  grad_shape_vel[k] = fe_values[extractors.velocities].gradient(k, q);
                  shape_div_vel[k]  = fe_values[extractors.velocities].divergence(k, q);
                  shape_pressure[k] = fe_values[extractors.pressure].value(k, q);

                  shape_inv_rho_theta[k] = inv_rho_partial_phi_ts * shape_phi_theta[k]
                      + inv_rho_partial_psi_ts * shape_psi_theta[k];
                  shape_rho_theta[k]     = rho_partial_phi_ts * shape_phi_theta[k]
                      + rho_partial_psi_ts * shape_psi_theta[k];

                  shape_inv_rho_partial_phi_theta[k] = inv_rho_partial2_phi2_ts * shape_phi_theta[k]
                      + inv_rho_partial_phi_partial_psi * shape_psi_theta[k];
                  shape_inv_rho_partial_psi_theta[k] = inv_rho_partial_phi_partial_psi * shape_phi_theta[k]
                      + inv_rho_partial2_psi2_ts * shape_psi_theta[k];

                  shape_h_ts[k] = shape_rho_theta[k] * material_derivative_ts
                      + rho_ts * (shape_phi[k]/present_timestep + shape_vel_theta[k] * grad_phi_ts
                                  + vel_ts * grad_shape_phi_theta[k]);

                  grad_shape_inv_rho_theta[k] = shape_inv_rho_partial_phi_theta[k] * grad_phi_ts
                      + shape_inv_rho_partial_psi_theta[k] * grad_psi_ts
                      + inv_rho_partial_phi_ts * grad_shape_phi_theta[k]
                      + inv_rho_partial_psi_ts * grad_shape_psi_theta[k];

                  shape_c_theta[k] = c_partial_phi_ts * shape_phi_theta[k]
                      + c_partial_psi_ts * shape_psi_theta[k];
                  shape_c_partial_phi_theta[k] = c_partial2_phi2_ts * shape_phi_theta[k]
                      + c_partial_phi_partial_psi_ts * shape_psi_theta[k];
                  shape_c_partial_psi_theta[k] = c_partial_phi_partial_psi_ts * shape_phi_theta[k]
                      + c_partial2_psi2_ts * shape_psi_theta[k];

                  shape_eta_theta[k] = eta_partial_phi_ts * shape_phi_theta[k]
                      + eta_partial_psi_ts * shape_psi_theta[k];

                  shape_e_theta = theta * (grad_shape_vel[k] + transpose(grad_shape_vel[k]) + 2./3. * shape_div_vel[k] * id_tensor);


                }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        double mat = 0;
                        double rhs = 0;

                        // w1, 2
//                        mat +=
//                        rhs +=

                        cell_matrix(i, j) +=
                            (viscosity *
                             scalar_product(grad_phi_u[i], grad_phi_u[j]) -
                             div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
                            fe_values.JxW(q);

                        cell_matrix2(i, j) += 1.0 / viscosity * phi_p[i] *
                                              phi_p[j] * fe_values.JxW(q);
                    }

                    const unsigned int component_i =
                        fe.system_to_component_index(i).first;
                    cell_rhs(i) += fe_values.shape_value(i, q) * fe_values.JxW(q);
                }
            }


            cell->get_dof_indices(local_dof_indices);
            constraints.distribute_local_to_global(cell_matrix,
                                                   cell_rhs,
                                                   local_dof_indices,
                                                   system_matrix,
                                                   system_rhs);

            constraints.distribute_local_to_global(cell_matrix2,
                                                   local_dof_indices,
                                                   preconditioner_matrix);
        }

    system_matrix.compress(VectorOperation::add);
    preconditioner_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
}



template <int dim>
void StokesProblem<dim>::solve()
{
    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::PreconditionAMG prec_A;
    {
        LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
        data.symmetric_operator = true;
#endif
        prec_A.initialize(system_matrix.block(0, 0), data);
    }

    LA::MPI::PreconditionAMG prec_S;
    {
        LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
        data.symmetric_operator = true;
#endif
        prec_S.initialize(preconditioner_matrix.block(1, 1), data);
    }

    using mp_inverse_t = LinearSolvers::InverseMatrix<LA::MPI::SparseMatrix,
          LA::MPI::PreconditionAMG>;
    const mp_inverse_t mp_inverse(preconditioner_matrix.block(1, 1), prec_S);

    const LinearSolvers::BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG,
          mp_inverse_t>
          preconditioner(prec_A, mp_inverse);

    SolverControl solver_control(system_matrix.m(),
                                 1e-10 * system_rhs.l2_norm());

    SolverMinRes<LA::MPI::BlockVector> solver(solver_control);

    LA::MPI::BlockVector distributed_solution(owned_partitioning,
            mpi_communicator);

    constraints.set_zero(distributed_solution);

    solver.solve(system_matrix,
                 distributed_solution,
                 system_rhs,
                 preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(distributed_solution);

    locally_relevant_solution = distributed_solution;
    const double mean_pressure =
        VectorTools::compute_mean_value(dof_handler,
                                        QGauss<dim>(velocity_degree + 2),
                                        locally_relevant_solution,
                                        dim);
    distributed_solution.block(1).add(-mean_pressure);
    locally_relevant_solution.block(1) = distributed_solution.block(1);
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


    {
        pcout << "n refinement " << n_refinement << ':' << std::endl;


        make_grid();


        setup_system();
        setup_initial_condition();

//        assemble_system();
//        solve();

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
        {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(n_refinement);
        }

        computing_timer.print_summary();
        computing_timer.reset();

        pcout << std::endl;
    }
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

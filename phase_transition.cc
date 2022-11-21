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
 * Author: Jiaqi Zhang, Clemson University, 2022
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

#define FORCE_USE_OF_TRILINOS
#define USE_DIRECT_SOLVER // direct solver cannot be used with block matrix
// #define USE_UMFPACK
#define USE_AXISYMMETRY // axisymmetric implementation
// #define USE_NEW_R

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
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
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
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <functional>

namespace Step55
{
using namespace dealii;

  enum class TestCase
  {
    test1,
    test2,
    test3
  };
  static const char *enum_str[] = {"test1", "test2", "test3"};

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
  double new_r(const double phi, const double alpha)
  {
    const auto f = [alpha](const double x) {
      const double alpha_2 = alpha * alpha;
      const double alpha_3 = alpha * alpha_2;
      const double x_3 = x * x * x;
      return x_3/alpha_2 - 2.*x_3*(x-alpha)/alpha_3 + 3.*x_3*(x-alpha)*(x-alpha)/(alpha_2*alpha_2);
    };
    if(phi<=0.)
      return 0.;
    else if(phi>=1.)
      return 1.;
    else if(phi<alpha && phi>0)
      return f(phi);
    else if(phi>=alpha && phi<=1.-alpha)
      return phi;
    else if(phi>1.-alpha && phi<1.)
      return 1. - f(1.-phi);
    else
    {
      Assert(false, ExcNotImplemented("inlinefunction new_r gets called with invalid phi"));
      return 0.;
    }
  }

  inline
  double new_r_prime(const double phi, const double alpha)
  {
    const auto f_prime = [&](const double x) {
      const double x_2 = x * x;
      return x_2 * (15.*x_2 - 32.*x + 18.);
    };
    if(phi<=0. || phi>=1.)
      return 0.;
    else if(phi<alpha && phi>0)
      return f_prime(phi/alpha);
    else if(phi>=alpha && phi<=1.-alpha)
      return 1;
    else if(phi>1.-alpha && phi<1.)
      return f_prime((1.-phi)/alpha);
    else
    {
      Assert(false, ExcNotImplemented("inlinefunction new_r_prime gets called with invalid phi"));
      return 0.;
    }
  } 

  inline
  double new_r_prime_prime(const double phi, const double alpha)
  {
    const auto f_prime_prime = [alpha](const double x) {
      return 12. * x * (3. * alpha - 5. * x) * (alpha - x)/std::pow(alpha, 4.);
    };
    if(phi<=0. || phi>=1.)
      return 0.;
    else if(phi<alpha && phi>0)
      return f_prime_prime(phi);
    else if(phi>=alpha && phi<=1.-alpha)
      return 0;
    else if(phi>1.-alpha && phi<1.)
      return -f_prime_prime(1.-phi);
    else
    {
      Assert(false, ExcNotImplemented("inlinefunction new_r_prime_prime gets called with invalid phi"));
      return 0.;
    }
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

  inline
  double q_prime_prime(const double phi)
  {
    return  std::cos(numbers::PI * phi) * 0.5 * numbers::PI * numbers::PI;
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

  inline double
  compute_lambda_from_surface_tension(const double rho_plus,
                                      const double rho_minus,
                                      const double surface_tension,
                                      const double eps)
  {
    // for phi_ac, rho_plus = rho_l, rho_minus = rho_s
    // for phi_ch, rho_plus = rho_l, rho_minus = rho_g
    const double diff_rho = rho_minus - rho_plus;
    const double sum_rho = rho_plus + rho_minus;

    if(std::fabs(diff_rho) < 1e-8)
      return surface_tension * 3. * numbers::SQRT2 * eps / (sum_rho * 0.5);
    else
    {
      const double tmp = (rho_minus * rho_plus)
      * (diff_rho * sum_rho + 2. * rho_minus * rho_plus * std::log(rho_plus/rho_minus))
      / (eps * numbers::SQRT2 * diff_rho * diff_rho * diff_rho);
      Assert(std::fabs(tmp) > 1e-8, ExcMessage("Lambda is negative."));
      return surface_tension / tmp;

    }
  }

  inline double
  compute_lambda_from_h(const double rho1,
                        const double surface_tension,
                        const double eps,
                        const double h)
  {
    // h is computed from mathematica.
    return surface_tension * eps / (h * numbers::SQRT2 * rho1);
  }

  inline double
  compute_surface_tension_from_lambda(const double rho_plus,
                                      const double rho_minus,
                                      const double lambda,
                                      const double eps)
  {
    const double diff_rho = rho_minus - rho_plus;
    const double sum_rho = rho_plus + rho_minus;

    if(std::fabs(diff_rho) < 1e-8)
      return lambda * (sum_rho * 0.5) / (eps * 3. * numbers::SQRT2);
    else
    {
      const double tmp = (rho_minus * rho_plus)
      * (diff_rho * sum_rho + 2. * rho_minus * rho_plus * std::log(rho_plus/rho_minus))
      / (eps * numbers::SQRT2 * diff_rho * diff_rho * diff_rho);
      return tmp * lambda;

    }
  }

  inline void
  limit_phase_field_function(double &pf_function)
  {
    if (pf_function < 0.)
      pf_function = 0.;
    else if (pf_function > 1.)
      pf_function = 1.;
  }

} //namespace inline funcitons

bool
fexists(const std::string &filename)
{
  std::ifstream ifile(filename.c_str());

  // return whether construction of the input file has succeeded;
  // success requires the file to exist and to be readable
  return static_cast<bool>(ifile);
}

void
move_file(const std::string &old_name, const std::string &new_name)
{
  int error = system(("mv " + old_name + " " + new_name).c_str());

  // If the above call failed, e.g. because there is no command-line
  // available, try with internal functions.
  if (error != 0)
    {
      if (fexists(new_name))
        {
          error = remove(new_name.c_str());
          AssertThrow(error == 0,
                      ExcMessage(
                        std::string("Unable to remove file: " + new_name +
                                    ", although it seems to exist. " +
                                    "The error code is " +
                                    Utilities::to_string(error) + ".")));
        }

      error = rename(old_name.c_str(), new_name.c_str());
      AssertThrow(error == 0,
                  ExcMessage(std::string("Unable to rename files: ") +
                             old_name + " -> " + new_name +
                             ". The error code is " +
                             Utilities::to_string(error) + "."));
    }
}


  template <int dim>
  void
  print_mesh_info(const Triangulation<dim> &triangulation)
  {
    ConditionalOStream pcout(
      std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
    pcout << "Mesh info:" << std::endl
              << " dimension: " << dim << std::endl
              << " no. of cells: " << Utilities::MPI::sum(triangulation.n_active_cells(), MPI_COMM_WORLD)
              << std::endl;

    {
      std::map<types::boundary_id, unsigned int> boundary_count;
      for (const auto &face : triangulation.active_face_iterators())
        if (face->at_boundary())
          boundary_count[face->boundary_id()]++;
      pcout << " boundary indicators: ";
      for (const std::pair<const types::boundary_id, unsigned int> &pair :
           boundary_count)
        {
          pcout << pair.first << "(" << Utilities::MPI::sum(pair.second, MPI_COMM_WORLD) << " times) ";
        }
      pcout << std::endl;
    }
    // std::ofstream out(filename);
    // GridOut       grid_out;
    // grid_out.write_vtu(triangulation, out);
    // pcout << " written to " << filename << std::endl << std::endl;
  }

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


  /******************************************************************
    This function labels the interface cells as refine and 
    non-interface cells as coarsen.
    1. The refinement is performed only if the mesh size is in [min_h,max_h]
       AND the level of refinement is in [min_level,max_level].
    2. The interface is defined as the zero level set of the component-th
       component of the vector solution. component is in [0,n_component)
       Apr 1, 2016, Pengtao Yue
    3. function returns true if a coarsen_and_refine is required and 
       false is the old mesh is to be kept.
  *******************************************************************/
  template <int dim, typename VectorType>
  bool label_mesh(const DoFHandler<dim>    & dof_handler, 
                  const FESystem<dim>      & fe,
                  const VectorType         & solution,
                  //minimum mesh size, defined on the interface
                  const double             & min_h,
                  //maximum mesh size, defined in the bulk
                  const double             & max_h,
                  //minimum refinement level
                  const int                & min_level,     
                  //maximum refinement level                                        
                  const int                & max_level,     
                  //component extractor for phase field phi
                  const Extractors<dim>    &extractors,
                  const double             &width = 0.48
                  )

  {
    const QTrapezoid<dim> quadrature;
    FEValues<dim> fe_values(fe,quadrature,update_values);
    const unsigned int n_q_points=quadrature.size();
    
    std::vector<double> phi_ch_values(n_q_points);
    std::vector<double> psi_ac_values(n_q_points);
    
    bool refine=false;

    const auto find_max_min_values = [&](const std::vector<double> &values) {
      double max_value = -std::numeric_limits<double>::max();
      double min_value = std::numeric_limits<double>::max();
      for (unsigned int q_point = 0; q_point < values.size(); ++q_point)
        {
          max_value = std::max(max_value, values[q_point]);
          min_value = std::min(min_value, values[q_point]);
        }

      return std::make_pair(min_value, max_value);
    };

    const auto is_interface_cell = [&](const double min_value,
                                       const double max_value) {
      return ((max_value - 0.5) * (min_value - 0.5) < 0 ||
              std::abs(max_value - 0.5) < width ||
              std::abs(min_value - 0.5) < width);
    };

    for(const auto &dof_cell : dof_handler.active_cell_iterators())
      if(dof_cell->is_locally_owned())
        {
          fe_values.reinit(dof_cell);
          fe_values[extractors.phi_ch].get_function_values(solution,
                                                           phi_ch_values);
          fe_values[extractors.psi_ac].get_function_values(solution,
                                                           psi_ac_values);

          const auto max_min_phi_ch = find_max_min_values(phi_ch_values);
          const auto max_min_psi_ac = find_max_min_values(psi_ac_values);

          const double phi_ch_min = max_min_phi_ch.first;
          const double phi_ch_max = max_min_phi_ch.second;
          const double psi_ac_min = max_min_psi_ac.first * phi_ch_max;
          const double psi_ac_max = max_min_psi_ac.second * phi_ch_max;

          const int    level = dof_cell->level();
          const double size  = dof_cell->minimum_vertex_distance();

          if (
            is_interface_cell(phi_ch_min, phi_ch_max) 
          ||
              is_interface_cell(psi_ac_min, psi_ac_max)
              )
            {
              // This is an interface cell
              // set_refine_flag() and set_coarsen_flag() can be found in calss
              // CellAccessor level() is inherited from class TriaAccessorBase
              if (level < max_level && size > min_h)
                {
                  dof_cell->clear_coarsen_flag();
                  dof_cell->set_refine_flag();
                  refine = true;
                }
            }
          else
            {
              if (level > min_level && size < 0.7 * max_h)
                {
                  // coarsening is of lower priority than refining. The mesh
                  // won't be corsened if the neighbor is already of finer size.
                  dof_cell->clear_refine_flag();
                  dof_cell->set_coarsen_flag();
                }
              else if (size > 1.5 * max_h)
                {
                  dof_cell->clear_coarsen_flag();
                  dof_cell->set_refine_flag();
                  refine = true;
                }
            }
        }

    return refine;
  }


  double
  transition_function(const double phi,
                      const double lower_bound,
                      const double upper_bound)
  {
    // if |phi| <= lower_bound, return 1
    // if |phi| > upper_bound, return 0
    // else layer
    const double abs_phi = std::fabs(phi);
    if (abs_phi <= lower_bound)
      return 1.;
    else if (abs_phi >= upper_bound)
      return 0.;
    else
      {
        const double x =
          (std::fabs(phi) - lower_bound) / (upper_bound - lower_bound);
        return (1 + 2 * x) * (1 - x) * (1 - x);
      }
}

    

namespace InitialConditions
{

  template <int dim>
  class InitialTemperature : public Function<dim>
  {
  public:
    InitialTemperature(const double   initial_t,
                          const Extractors<dim> &ex)
      : Function<dim>(dim + 6)
      , initial_t(initial_t)
      , extractors(ex)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    private:
    const double initial_t;
    const Extractors<dim> &extractors;
  };

  template <int dim>
  double
  InitialTemperature<dim>::value(const Point<dim>  &,
                                 const unsigned int comp) const
  {
    if(comp == extractors.temperature.component)
      return initial_t;
  }


  template <int dim>
  class InitialValues : public Function<dim>
  {
  public:
      InitialValues (const double epsilon, 
                     const double initial_t, 
                     const double melting_t,
                     const TestCase testcase, 
                     const Extractors<dim> &ex)
                  : Function<dim>(dim + 6),
                    eps(epsilon),
                    initial_temperature(initial_t),
                    melting_temperature(melting_t),
                    test_case(testcase),
                    extractors(ex)
                    {}

      virtual void vector_value(const Point<dim> &p,
                                Vector<double> &  value) const override;

      double get_alpha() const
      {
        return alpha;
      }

      double get_boundary_temperature() const
      {
        return boundary_temperature;
      }

      void set_boundary_temperature(const double t)
      {
        boundary_temperature = t;
      }
      
  private:
      const double eps;
      const double initial_temperature;
      const double melting_temperature;
      double boundary_temperature = -1.0;
      const TestCase test_case;
      const Extractors<dim> extractors;
      const double alpha = 0.2;

  };

  template <int dim>
  void InitialValues<dim>::vector_value(const Point<dim> &p,
                                        Vector<double> &  values) const
  {
      const double eps1=eps * numbers::SQRT2;  //sqrt(2)* eps
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
            const double w_ac = 0.5;
            const double d = x - w_ac;
            const double psi = 0.5 * (1. + std::tanh(d/eps1));

            const double w_ch = 0.8;
            const double d_ch = w_ch - x;
            const double phi = 0.5 * (1. + std::tanh(d_ch/eps1));

            for(unsigned int comp = 0; comp < values.size(); ++comp)
              {
                if (comp == extractors.psi_ac.component)
                  values(comp) = psi;
                else if (comp == extractors.temperature.component)
                  values(comp) = melting_temperature * (0.9 + 0.1 * x);
                else if (comp == extractors.phi_ch.component)
                  values(comp) = phi;
                else
                  values(comp) = 0;

              }          

            break;
          }
        case TestCase::test3: {
            Assert(boundary_temperature > 0, ExcMessage("Boundary temperature not set!"));
            Point<dim> center;
            const double R = 1.;
            double r;
            switch (dim)
              {
              case 2: //2D case
                {
                  const double height = 0.72455039792 * R;
                  // const double height = R;
                  const double center_y = R - height;
                  center = Point<dim>(0, -center_y);
                  r = p.distance(center);
                  break;
                }
              case 3: //3D case
                {
                  ExcNotImplemented("3D not implemented yet!");
                  break;
                }
              }

            const double d = R - r;
            const double phi = 0.5 * (1. + std::tanh(d/eps1));

            const double initial_solid_layer = 0.08; // has to below melting temperature
            const double psi = 0.5 * (1. + std::tanh((y - initial_solid_layer)/eps1));

            const double temperature_transition_function = transition_function(y, initial_solid_layer + 0.1, initial_solid_layer+0.2);

            const double initial_t =
              (1. - temperature_transition_function) * initial_temperature +
              temperature_transition_function * melting_temperature;

            for(unsigned int comp = 0; comp < values.size(); ++comp)
              {
                if (comp == extractors.psi_ac.component)
                  values(comp) = psi;
                else if (comp == extractors.temperature.component)
                  values(comp) = initial_t;
                else if (comp == extractors.phi_ch.component)
                  values(comp) = phi;
                else
                  values(comp) = 0;

              }
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


namespace DimensionlessGroups
{
  // const double G      =2.000000e+01; // 1.159722e+06; // 1/G characterises Thompson-Gibbs effect
  // const double Pi_T   =8.000e+00; //  1.452778e+04; // sensible heat / surface tension (AC/CH)
  // const double Pi_eta =1.6e+01;//  1.090104e+06; // sensible heat / visicosity
  // const double Ste    =4.000e-0;  //  1.252695e-02; // Stefan number
  // const double We     =1.250000e-01;  //  1.000000e+00; // Weber number
  // const double Re     =2.500000e-01;  //  7.503584e+01; // Reynolds number
  // const double Pe     =1.000000e+00;  //  1.010063e+03; // Peclet number

  
  const double G      = 1.159722e+03; // 1/G characterises Thompson-Gibbs effect
  const double Pi_T   = 1.452778e+04; // sensible heat / surface tension (AC/CH)
  const double Pi_eta = 1.090104e+06; // sensible heat / visicosity
  const double Ste    = 1.252695e+01; // Stefan number
  const double We     = 1.000000e+00; // Weber number
  const double Re     = 7.503584e+01; // Reynolds number
  const double Pe     = 1.010063e+0; // Peclet number
  const double G_ch      = 0; // 1/G characterises Thompson-Gibbs effect
  const double Pi_T_ch   = 0; // sensible heat / surface tension (AC/CH)

  // Yue's parameters
  // const double G      = 1.159722e+03; // 1/G characterises Thompson-Gibbs effect
  // const double Pi_T   = 1.452778e+04; // sensible heat / surface tension (AC/CH)
  // const double Pi_eta = 1.090104e+06; // sensible heat / visicosity
  // const double Ste    = 1.252695e+01; // Stefan number
  // const double We     = 1.000000e+00; // Weber number
  // const double Re     = 7.503584e+01; // Reynolds number
  // const double Pe     = 1.010063e+03; // Peclet number

  // exp parameters
  // const double G      = 1.159722e+06; // 1/G characterises Thompson-Gibbs effect
  // const double Pi_T   = 1.452778e+04; // sensible heat / surface tension (AC/CH)
  // const double Pi_eta = 1.090104e+06; // sensible heat / visicosity
  // const double Ste    = 1.252695e-02; // Stefan number
  // const double We     = 1.000000e+00; // Weber number
  // const double Re     = 7.503584e+01; // Reynolds number
  // const double Pe     = 1.010063e+03; // Peclet number

  const double one_over_Pe = 1.0/Pe;
  const double one_over_Pi_T = 1.0 / Pi_T;
  const double one_over_Re = 1.0/Re;
  const double one_over_We = 1./ We; 
  const double one_over_Ste = 1.0 / Ste;
  const double one_over_Pi_eta = 1.0 / Pi_eta;
  void print_dimensionless_groups()
  {
    ConditionalOStream pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
    pcout<< " G: "<< G << std::endl
    << " Pi_T: "<< Pi_T << std::endl
    << " Pi_eta: "<< Pi_eta << std::endl
    << " Ste: "<< Ste << std::endl
    << " We: "<< We << std::endl
    << " Re: "<< Re << std::endl
    << " Pe: "<< Pe << std::endl;

  }
} // namespace DimensionlessGroups

template <int dim>
class StokesProblem
{
public:
    StokesProblem(unsigned int velocity_degree, const TestCase & testcase);

    void run();
    void test_adaptive_refinement();

private:
#ifdef USE_DIRECT_SOLVER
  #ifdef USE_UMFPACK
    using VectorType = Vector<double>;
    using MatrixType = SparseMatrix<double>;    
  #else
    using VectorType = LA::MPI::Vector;
    using MatrixType = LA::MPI::SparseMatrix;
  #endif
#else
    using VectorType = LA::MPI::BlockVector;
    using MatrixType = LA::MPI::BlockSparseMatrix;
#endif
    void create_coarse_grid(parallel::distributed::Triangulation<dim> &coarse_grid) const ;
    void set_boundary_ids(parallel::distributed::Triangulation<dim> &coarse_grid) const ;
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
    void print_variables() const;

    std::vector<const FiniteElement<dim> *>
    create_fe_list(const unsigned int velocity_degree);

    std::vector<unsigned int> create_fe_multiplicities();

    void
    save_checkpoint(const unsigned int step_number, const double runtime);
    void
    load_checkpoint(unsigned int &step_number, double &runtime);



    const unsigned int velocity_degree;
    const unsigned int quadrature_degree;

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

    const double density_l, density_s, density_g;
    const double product_density_g_c_g;  // experiment: rho_g*c_g
    const double inv_density_s, inv_density_l, inv_density_g;
    double present_timestep, old_timestep, fix_timestep;
    const double cl,cs,cg;
    const double eta_l,eta_s,eta_g;
    const double surface_tension_phi_ch;
    const double surface_tension_psi_ac;
    const double lambda_phi, lambda_psi;
    const double mobility_phi, mobility_psi;
    const double latent_heat, melting_t;
    const double k_l; // thermal_conductivity
    const double k_s;
    const double k_g;
    const double initial_temperature; //Ta
    const double boundary_temperature; 
    const double ambient_pressure;

    const MappingQ<dim> mapping;

    double hmin;

    const unsigned int wall_boundary_id = 11;

    const double static_contact_angle; // theta_s
    const double one_over_wall_relaxation_gamma;
    const Tensor<1,dim> wall_velocity;
    const bool use_adaptive_refinement;

    const double min_mesh_size;
    const double max_mesh_size;   

    std::string output_dir;
    std::string checkpoints_dir;

    bool relax_phase_field = false;

    unsigned int step_number = 0;
    double       runtime     = 0.;
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
StokesProblem<dim>::StokesProblem(unsigned int    velocity_degree,
                                  const TestCase &testcase)
  : velocity_degree(velocity_degree)
  , quadrature_degree(2 * velocity_degree + 4)
  , mpi_communicator(MPI_COMM_WORLD)
  , fe(create_fe_list(velocity_degree), create_fe_multiplicities())
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening),
                    parallel::distributed::Triangulation<dim>::mesh_reconstruction_after_repartitioning)
  , dof_handler(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , component_ids(ComponentIndices<dim>())
  , extractors(ComponentIndices<dim>())
  , eps(0.02)
  , test_case(testcase)
  , n_refinement(7)
  , density_l(1.)  
  , density_s(9.162000e-01 * density_l)
  , density_g(0.01 * density_l)
  , product_density_g_c_g(3.106963e-04)
  , inv_density_s(1. / density_s)
  , inv_density_l(1. / density_l)
  , inv_density_g(1. / density_g)
  , present_timestep(1.5e-3)
  , old_timestep(present_timestep)
  , fix_timestep(present_timestep)
  , cl(1.)
  , cs(cl)//(4.899618e-01)
  , cg(cl)//(2.404398e-01)
  , eta_l(1.)
  , eta_s(200.)
  , eta_g(9.670022e-03)
  , surface_tension_phi_ch(1.)
  , surface_tension_psi_ac(0.6)
  , lambda_phi(InlineFunctions::compute_lambda_from_h(density_l, surface_tension_phi_ch, eps, 0.0115298)) // surface tension formula is changed, need to compute the factor.
  , lambda_psi(InlineFunctions::compute_lambda_from_h(density_l, surface_tension_psi_ac, eps, 0.159505)) /*which density should be used? for water ice  h= 0.159505, g=0.1594389483*/ 
  , mobility_phi(1e-3)
  , mobility_psi(1.)
  , latent_heat(1)
  , melting_t(273.)
  , k_l(1.) //  thermal_conductivity(1.)
  , k_s(3.994602e+00)
  , k_g(4.383266e-02)
  , initial_temperature(melting_t+20.)
  , boundary_temperature(melting_t-2.)
  , ambient_pressure(0.)
  , mapping(1)
  , static_contact_angle(numbers::PI * 73.47127431/180.) // 73.47 degrees
  , one_over_wall_relaxation_gamma(100)
  , wall_velocity(Tensor<1, dim>())
  , use_adaptive_refinement(true)
  , min_mesh_size(0.01)
  , max_mesh_size(0.5)
{
  print_variables();
  DimensionlessGroups::print_dimensionless_groups();
}

template <int dim>
void StokesProblem<dim>:: create_coarse_grid(parallel::distributed::Triangulation<dim> &coarse_grid) const 
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
        GridGenerator::hyper_cube(coarse_grid, 0., width, colorize);
        break;
      }
    case TestCase::test2: {
        AssertDimension(dim, 2);
        const Point<dim> p0;
        const Point<dim> p1 = Point<dim>(2., 1./4.);

        std::vector< unsigned int > repetitions(dim,1); 
        repetitions[0] = 8;

        const bool   colorize = true;
        GridGenerator::subdivided_hyper_rectangle(
          coarse_grid, repetitions, p0, p1, colorize);
        break;
      }
    case TestCase::test3: {
//        If the colorize flag is true,
//        then the boundary_ids of the boundary faces are assigned,
//            such that the lower one in x-direction is 0, the upper one is 1.
//            The indicators for the surfaces in y-direction are 2 and 3,
//            the ones for z are 4 and 5.
        const bool   colorize = true;
        const double width = 2.;
        GridGenerator::hyper_cube(coarse_grid, 0., width, colorize);
        // // note that some boundary ids are not saved in the checkpoints.
        // set_boundary_ids();
        break;
      }      
    default:
      Assert(false, ExcNotImplemented("Setting up iniital grid: Please choose the right test case"));
    }  
    coarse_grid.signals.post_refinement.connect(
    [this, &coarse_grid]()
    {
      this->set_boundary_ids(coarse_grid);
    });
}

template <int dim>
void StokesProblem<dim>::make_grid()
{
  create_coarse_grid(triangulation);

  switch  (test_case)
    {
    case TestCase::test1: {
//        If the colorize flag is true,
//        then the boundary_ids of the boundary faces are assigned,
//            such that the lower one in x-direction is 0, the upper one is 1.
//            The indicators for the surfaces in y-direction are 2 and 3,
//            the ones for z are 4 and 5.
        triangulation.refine_global(n_refinement);

        break;
      }
    case TestCase::test2: {
        AssertDimension(dim, 2);

        if(use_adaptive_refinement)
          triangulation.refine_global(3);
        else
          triangulation.refine_global(n_refinement);
        break;
      }
    case TestCase::test3: {

        if(use_adaptive_refinement)
          triangulation.refine_global(3);
        else
          triangulation.refine_global(n_refinement);

        break;
      }      
    default:
      Assert(false, ExcNotImplemented("Setting up iniital grid: Please choose the right test case"));
    }
                                         
  const unsigned int max_refinement_level = n_refinement;
  if (use_adaptive_refinement)
    {
      for (unsigned int i = 0; i < max_refinement_level; ++i)
        {
          // dof_handler is required to apply initial condition
          dof_handler.distribute_dofs(fe);

          { // make hanging node constraints, used in setting up initial
            // condition
            constraints_boundary.clear();
            DoFTools::make_hanging_node_constraints(dof_handler,
                                                    constraints_boundary);
            constraints_boundary.close();
          }

          // initialize the solution vector
          locally_relevant_dofs.clear();
          DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                  locally_relevant_dofs);
          locally_relevant_solution.reinit(dof_handler.locally_owned_dofs(),
                                           locally_relevant_dofs,
                                           mpi_communicator);
          setup_initial_condition();

          const bool refine_mesh_local =
            label_mesh<dim, VectorType>(dof_handler,
                                        fe,
                                        locally_relevant_solution,
                                        min_mesh_size,
                                        max_mesh_size,
                                        0,
                                        n_refinement,
                                        extractors);

          const bool refine_mesh =
            Utilities::MPI::logical_or(refine_mesh_local, mpi_communicator);

          if (refine_mesh)
            {
              triangulation.execute_coarsening_and_refinement();

              pcout << "   Refinement level: " << i << std::endl
                        << "   Number of active cells: "
                        << triangulation.n_active_cells() << std::endl
                        << "   Total number of cells: "
                        << triangulation.n_cells() << std::endl;
              const double hmin =
                GridTools::minimal_cell_diameter(triangulation) /
                std::sqrt(1. * dim);
              pcout << "hmin = " << hmin << std::endl;
            }
          else
            {
              break;
            }
        }
    }

  print_mesh_info(triangulation);
  hmin = GridTools::minimal_cell_diameter(triangulation, mapping)/std::sqrt(dim*1.);
  pcout<<" hmin = "<<hmin<<std::endl;
}

template <int dim>
void StokesProblem<dim>::
set_boundary_ids(parallel::distributed::Triangulation<dim> &coarse_grid) const
{
switch  (test_case)
    {
    case TestCase::test1: {
        break;
      }
    case TestCase::test2: {
        break;
      }
    case TestCase::test3: {
        for(auto &cell : coarse_grid.active_cell_iterators())
          for(const auto &f : cell->face_indices())
            if(cell->face(f)->at_boundary())
              if(cell->face(f)->boundary_id() == 2)
                cell->face(f)->set_all_boundary_ids(wall_boundary_id);
        break;
      }      
    default:
      Assert(false, ExcNotImplemented("Setting up iniital grid: Please choose the right test case"));
    }  

}

template <int dim>
void StokesProblem<dim>::setup_initial_condition()
{
  VectorType tmp_initial_sol;
#ifdef USE_DIRECT_SOLVER
  #ifdef USE_UMFPACK
    tmp_initial_sol.reinit(dof_handler.locally_owned_dofs());
  #else
    tmp_initial_sol.reinit(dof_handler.locally_owned_dofs(), mpi_communicator);
  #endif
#else
  tmp_initial_sol.reinit(block_owned_partitioning, mpi_communicator);
#endif

  InitialConditions::InitialValues<dim> initial_values(
    eps, initial_temperature, melting_t, test_case, extractors);

  switch (test_case)
    {
    case TestCase::test1: {
      
        break;
      }
    case TestCase::test2: {
      
        break;
      }
    case TestCase::test3: {
        initial_values.set_boundary_temperature(boundary_temperature);
        break;
      }
    default:
      Assert(false, ExcNotImplemented("Setting up iniital condition: Please choose the right test case"));
    }

  VectorTools::interpolate(dof_handler,
                           initial_values,
                           tmp_initial_sol);

  constraints_boundary.distribute(tmp_initial_sol);

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
          // T = Ta
          const auto               const_function =
            Functions::ConstantFunction<dim>(initial_temperature, fe.n_components());
          const auto               zero_function =
            Functions::ZeroFunction<dim>(fe.n_components());
          
          const std::map<types::boundary_id, const Function<dim> *>
            zero_function_map = {{1, &zero_function}, {3, &zero_function}};
          const std::map<types::boundary_id, const Function<dim> *>
            const_function_map = {{1, &const_function}, {3, &const_function}};            

          ComponentMask temperature_masked(fe.n_components(), false);
          temperature_masked.set(extractors.temperature.component, true);
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   const_function_map,
                                                   constraints_boundary,
                                                   temperature_masked);

          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   zero_function_map,
                                                   constraints_newton_update,
                                                   temperature_masked);
        }
        break;
      }
      case TestCase::test2: {
        // const double alpha =
        //   InitialConditions::InitialValues<dim>(
        //     eps, initial_temperature, melting_t, test_case, extractors)
        //     .get_alpha();
        {
          // x=0, boundary id=0,
          // velocity (u,v,w) = 0 at id = 0
          // T = (1-alpha) * TM at x=0, bc id = 0

          const types::boundary_id bc_id = 0;
          ComponentMask            vel_masked(fe.n_components(), false);
          for (unsigned int d = 0; d < dim; ++d)
            vel_masked.set(extractors.velocities.first_vector_component + d,
                           true);

          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(
                                                     fe.n_components()),
                                                   constraints_boundary,
                                                   vel_masked);

          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(
                                                     fe.n_components()),
                                                   constraints_newton_update,
                                                   vel_masked);

          // temperature constraints
          ComponentMask temperature_masked(fe.n_components(), false);
          temperature_masked.set(extractors.temperature.component, true);
          const double t_alpha = 0.9 * melting_t;

          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler,
            bc_id,
            Functions::ConstantFunction<dim>(t_alpha, fe.n_components()),
            constraints_boundary,
            temperature_masked);

          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(
                                                     fe.n_components()),
                                                   constraints_newton_update,
                                                   temperature_masked);
        }

        {
          // x=2, boundary id= 1, 
          // v = 0 at id = 1
          // T = (1+alpha) * TM at x=2, bc id = 1
          const types::boundary_id bc_id = 1;
          ComponentMask            vel_v_masked(fe.n_components(), false);
            vel_v_masked.set(extractors.velocities.first_vector_component + 1,
                           true);

          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(
                                                     fe.n_components()),
                                                   constraints_boundary,
                                                   vel_v_masked);

          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(
                                                     fe.n_components()),
                                                   constraints_newton_update,
                                                   vel_v_masked);

          // temperature constraints
          ComponentMask temperature_masked(fe.n_components(), false);
          temperature_masked.set(extractors.temperature.component, true);
          const double t_alpha = melting_t;

          VectorTools::interpolate_boundary_values(
            mapping,
            dof_handler,
            bc_id,
            Functions::ConstantFunction<dim>(t_alpha, fe.n_components()),
            constraints_boundary,
            temperature_masked);

          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(
                                                     fe.n_components()),
                                                   constraints_newton_update,
                                                   temperature_masked);          
        }

        {
          // y = 0,1, boundary id = 2, 3
          // v = 0, both
          // zero shear stress and zero heat flux (both embedded in the weak
          // form)
          ComponentMask vel_v_masked(fe.n_components(), false);
          vel_v_masked.set(extractors.velocities.first_vector_component + 1,
                           true);
          const auto zero_function =
            Functions::ZeroFunction<dim>(fe.n_components());
          const std::map<types::boundary_id, const Function<dim> *>
            zero_function_map = {{2, &zero_function}, {3, &zero_function}};
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   zero_function_map,
                                                   constraints_boundary,
                                                   vel_v_masked);
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   zero_function_map,
                                                   constraints_newton_update,
                                                   vel_v_masked);
        }
        break;
      }
    case TestCase::test3: {

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
          // y=0, boundary id=2, velocity (u,v,w)
          // velocity = 0
          // temperature = boundary_temperature
          // if you want to use wall relaxation, you need to set the
          // boundary id to wall_bc_id;
          const types::boundary_id bc_id = wall_boundary_id;
          ComponentMask vel_masked(fe.n_components(), false);
          for(unsigned int d = 0; d < dim; ++d)
            vel_masked.set(extractors.velocities.first_vector_component + d,
                           true);
          
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(fe.n_components()),
                                                   constraints_boundary,
                                                   vel_masked);

          VectorTools::interpolate_boundary_values(dof_handler,
                                                   bc_id,
                                                   Functions::ZeroFunction<dim>(fe.n_components()),
                                                   constraints_newton_update,
                                                   vel_masked);

          ComponentMask temperature_masked(fe.n_components(), false);
          temperature_masked.set(extractors.temperature.component, true);
          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc_id,
                                                    Functions::ConstantFunction<dim>(
                                                      boundary_temperature,
                                                      fe.n_components()),
                                                   constraints_boundary,
                                                   temperature_masked);
                                                             

          VectorTools::interpolate_boundary_values(mapping,
                                                   dof_handler,
                                                   bc_id,
                                                    Functions::ZeroFunction<dim>(
                                                      fe.n_components()),
                                                   constraints_newton_update,
                                                   temperature_masked);                                                   
        }

         {
          //boundary id=1, 3
          const types::boundary_id bc_id= 1;
          // zero shear stress and zero heat flux (both embedded in the weak form)
          // when solving CH + NS, we observe larege velocity on the boundary 1,
          // so use slip condition. 
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
                   || d == component_ids.mu_phi_ch
                   || d == component_ids.phi_ch
                   || d == component_ids.mu_psi_ac
                   || d == component_ids.psi_ac
                   || d == component_ids.pressure))
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
    {
      TimerOutput::Scope t(computing_timer, "renumbering");
      // DoFRenumbering::component_wise(dof_handler); // doesn't work
      DoFRenumbering::hierarchical(dof_handler);
      DoFRenumbering::Cuthill_McKee(dof_handler);
    }

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
    Timer timer(mpi_communicator);


    pcout<<" assemble system... "<<std::flush;

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
    const QGauss<dim-1> face_quadrature_formula(quadrature_degree);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    FEFaceValues<dim> fe_face_values(fe,
                                 face_quadrature_formula,
                                 update_values | update_quadrature_points |
                                  update_normal_vectors | update_JxW_values |
                                  update_gradients);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();

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

    std::vector<double> shape_pressure_reformulated_term(dofs_per_cell);

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

            fe_values[extractors.mu_psi_ac].get_function_values(current_solution, mu_psi_ac_star);
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
#ifdef USE_AXISYMMETRY
            const auto &quadrature_points = fe_values.get_quadrature_points();
#endif

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
              double jxwq = fe_values.JxW(q);
#ifdef USE_AXISYMMETRY
              jxwq *= quadrature_points[q][0];
              const double one_over_r = 1.0 / quadrature_points[q][0];
              
#endif
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

//              const double material_derivative_ts = ((phi_ch_star[q] - phi_ch_n[q])/present_timestep + vel_ts * grad_phi_ch_ts);
              const double material_derivative_ts = ((phi_ch_star[q] - phi_ch_n[q])/present_timestep + vel_bar * grad_phi_ch_ts);              
//              {
//                std::cout<<" phi_ch_star: "<<phi_ch_star[q]
//                           << " phi_ch_n: "<<phi_ch_n[q]
//                              << " present time step "<<present_timestep
//                              <<" vel_ts: "<< vel_ts
//                             <<" frad_phi_ch_ts: "<<grad_phi_ch_ts<<std::endl;
//              }

              const double h_ts   = rho_ts * material_derivative_ts;

              // const Tensor<1,dim> grad_inv_rho_ts = inv_rho_partial_phi_ch_ts * grad_phi_ch_ts + inv_rho_partial_psi_ac_ts * grad_psi_ac_ts;

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

              // k: the thermal conductivity
              const double k_ts = InlineFunctions::f(phi_ch_ts,
                                                     psi_ac_ts,
                                                     k_l,
                                                     k_s,
                                                     k_g); // thermal_conductivity
              const double k_partial_phi_ch_ts = InlineFunctions::f_partial_phi(phi_ch_ts,
                                                                             psi_ac_ts,
                                                                             k_l,
                                                                             k_s,
                                                                             k_g);
              const double k_partial_psi_ac_ts = InlineFunctions::f_partial_psi(phi_ch_ts,
                                                                              psi_ac_ts,
                                                                              k_l,
                                                                              k_s);

              // E_theta_star
              const Tensor<2,dim> e_ts = grad_vel_ts + transpose(grad_vel_ts) -  2./3. * div_vel_ts * id_tensor;
#ifndef USE_NEW_R
              const double r_phi_ch_ts = InlineFunctions::r(phi_ch_ts);
              const double r_psi_ac_ts = InlineFunctions::r(psi_ac_ts);
              const double r_prime_phi_ch_ts = InlineFunctions::r_prime(phi_ch_ts);
              const double r_prime_psi_ac_ts = InlineFunctions::r_prime(psi_ac_ts);
              const double r_prime_prime_phi_ch_ts = InlineFunctions::r_prime_prime(phi_ch_ts);
              const double r_prime_prime_psi_ac_ts = InlineFunctions::r_prime_prime(psi_ac_ts);
              const double artificial_diffusion_coefficient
              = InlineFunctions::r(0.2) - InlineFunctions::r(std::min(phi_ch_n[q], 0.2)); //0.1
#else
              const double r_alpha = 0.05;
              const double r_phi_ch_ts = InlineFunctions::new_r(phi_ch_ts, r_alpha);
              const double r_psi_ac_ts = InlineFunctions::new_r(psi_ac_ts, r_alpha);
              const double r_prime_phi_ch_ts = InlineFunctions::new_r_prime(phi_ch_ts, r_alpha);
              const double r_prime_psi_ac_ts = InlineFunctions::new_r_prime(psi_ac_ts, r_alpha);
              const double r_prime_prime_phi_ch_ts = InlineFunctions::new_r_prime_prime(phi_ch_ts, r_alpha);
              const double r_prime_prime_psi_ac_ts = InlineFunctions::new_r_prime_prime(psi_ac_ts, r_alpha);
              const double artificial_diffusion_coefficient
              = InlineFunctions::new_r(0.2, r_alpha) - InlineFunctions::new_r(std::min(phi_ch_n[q], 0.2), r_alpha);
#endif

              const double w_psi_ac_ts = InlineFunctions::w(psi_ac_ts, eps);
              const double w_prime_psi_ac_ts = InlineFunctions::w_prime(psi_ac_ts, eps);
              const double w_prime_phi_ch_ts = InlineFunctions::w_prime(phi_ch_ts, eps);
              const double w_prime_prime_phi_ch_ts = InlineFunctions::w_prime_prime(phi_ch_ts, eps);
              const double w_prime_prime_psi_ac_ts = InlineFunctions::w_prime_prime(psi_ac_ts, eps);

              const double pressure_reformulated_term = (lambda_phi * (grad_phi_ch_ts * grad_phi_ch_ts) 
                     + r_phi_ch_ts * lambda_psi * (grad_psi_ac_ts * grad_psi_ac_ts));
              const Tensor<2,dim> gamma_ts = lambda_phi * outer_product(grad_phi_ch_ts, grad_phi_ch_ts)
                  + r_phi_ch_ts * lambda_psi * outer_product(grad_psi_ac_ts, grad_psi_ac_ts)
                  - pressure_reformulated_term * id_tensor;

              const Tensor<1,dim> D_vel_Dt_ts = (vel_star[q] - vel_n[q])/present_timestep +  grad_vel_ts * vel_bar
                  + 0.5 * div_vel_bar * vel_ts;

//              const double D_temperature_Dt_ts = (temperature_star[q] - temperature_n[q])/present_timestep + vel_ts * grad_temperature_ts;
              const double D_temperature_Dt_ts = (temperature_star[q] - temperature_n[q])/present_timestep + vel_bar * grad_temperature_ts;

              const double log_t_ts_tm = std::log(temperature_ts/melting_t);
              Assert(numbers::is_finite(log_t_ts_tm), ExcMessage("log_t_ts_tm is not finite, which means temperature is negative"));
              // TM(1-T_ts/TM) + T_ts*log(T_ts/TM), used in  w3 and w5
              const double w35_reuse_term1 = melting_t * (1. - temperature_ts/melting_t) + temperature_ts * log_t_ts_tm;
              // w(phi_ch_ts) + 0.5 |grad_psi_ac_ts|^2
              const double w3_reuse_term2 = w_psi_ac_ts + 0.5 * (grad_psi_ac_ts * grad_psi_ac_ts);

              // (psi_ac_ts - psi_ac_n)/dt + vel_ts * grad_psi_ac_ts
//              const double D_psi_D_t_ts = (psi_ac_star[q] - psi_ac_n[q])/present_timestep + vel_ts * grad_psi_ac_ts;
              const double D_psi_D_t_ts = (psi_ac_star[q] - psi_ac_n[q])/present_timestep + vel_bar * grad_psi_ac_ts;

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
//                      + rho_ts * (shape_phi_ch[k]/present_timestep + shape_vel_theta[k] * grad_phi_ch_ts
//                                  + vel_ts * grad_shape_phi_ch_theta[k]);
                      + rho_ts * (shape_phi_ch[k]/present_timestep + vel_bar* grad_shape_phi_ch_theta[k]);
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
                  shape_pressure_reformulated_term[k] = (2. * lambda_phi * (grad_phi_ch_ts * grad_shape_phi_ch_theta[k])
                         + 2. * r_phi_ch_ts * lambda_psi * (grad_psi_ac_ts * grad_shape_psi_ac_theta[k])
                         + r_prime_phi_ch_ts * lambda_psi * shape_phi_ch_theta[k] * (grad_psi_ac_ts * grad_psi_ac_ts));
                  shape_gamma_theta[k] = lambda_phi * (grad_phi_ch_ts_grad_shape_phi_ch_theta
                                                       + transpose(grad_phi_ch_ts_grad_shape_phi_ch_theta))
                      + lambda_psi * (r_prime_phi_ch_ts * shape_phi_ch_theta[k] * outer_product(grad_psi_ac_ts, grad_psi_ac_ts)
                                      + r_phi_ch_ts * (grad_psi_ac_ts_grad_shape_psi_ac_theta + transpose(grad_psi_ac_ts_grad_shape_psi_ac_theta)))
                      - shape_pressure_reformulated_term[k] * id_tensor;
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  if(assemble_matrix)
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          // w1,2
                          mat = shape_h_ts[j] * shape_phi_ch[i] + mobility_phi * grad_shape_mu_phi_ch[j] * grad_shape_phi_ch[i];

                          // w3
                          //  term i
                          mat += shape_mu_phi_ch[j] * shape_mu_phi_ch[i];
                          // //  term ii
                          mat += (-((r_prime_psi_ac_ts * r_prime_phi_ch_ts * shape_psi_ac_theta[j]
                                     + r_psi_ac_ts * r_prime_prime_phi_ch_ts * shape_phi_ch_theta[j]
                                     ) * latent_heat * (1. - temperature_ts/melting_t)
                                    ) * DimensionlessGroups::G_ch
                                  + r_psi_ac_ts * r_prime_phi_ch_ts * latent_heat/melting_t * shape_temperature_theta[j] * DimensionlessGroups::G_ch
                                  + (shape_c_partial_phi_theta[j] * w35_reuse_term1
                                     + c_partial_phi_ch_ts * shape_temperature_theta[j] * log_t_ts_tm
                                     ) * DimensionlessGroups::Pi_T_ch
                                  ) * shape_mu_phi_ch[i];
                          //  term iii
                          mat += (-(lambda_phi * w_prime_prime_phi_ch_ts
                                    + lambda_psi * r_prime_prime_phi_ch_ts * w3_reuse_term2
                                    ) * shape_phi_ch_theta[j]
                                  - lambda_psi * r_prime_phi_ch_ts * (w_prime_psi_ac_ts * shape_psi_ac_theta[j] + grad_psi_ac_ts * grad_shape_psi_ac_theta[j])
                                  - (shape_pressure[j] * inv_rho_partial_phi_ch_ts + pressure_star[q] * shape_inv_rho_partial_phi_theta[j]
                                     ) * DimensionlessGroups::We
                                  ) * shape_mu_phi_ch[i];
                          //  term iv
                          mat += - lambda_phi * (grad_shape_phi_ch_theta[j] * grad_shape_mu_phi_ch[i]);
                              // -  lambda_phi * shape_rho_theta[j] * (grad_phi_ch_ts * grad_inv_rho_ts) * shape_mu_phi_ch[i]
                              // -  lambda_phi * rho_ts * (grad_shape_phi_ch_theta[j] * grad_inv_rho_ts) * shape_mu_phi_ch[i]
                              // -  lambda_phi * rho_ts * (grad_phi_ch_ts * grad_shape_inv_rho_theta[j]) * shape_mu_phi_ch[i];

                          // todo: add face integral term v

                          // w4
//                          mat += (rho_ts * (shape_psi_ac[j]/present_timestep + vel_ts * grad_shape_psi_ac_theta[j]
//                                            + shape_vel_theta[j] * grad_psi_ac_ts)
                          mat += (rho_ts * (shape_psi_ac[j]/present_timestep + vel_bar * grad_shape_psi_ac_theta[j])
                                  + shape_rho_theta[j] * D_psi_D_t_ts + mobility_psi * shape_mu_psi_ac[j]) * shape_psi_ac[i];

                          // w5
                          //  term i
                          mat += shape_mu_psi_ac[j] * shape_mu_psi_ac[i];
                          //  term ii
                          mat += ( -( (r_prime_prime_psi_ac_ts * r_phi_ch_ts * shape_psi_ac_theta[j]
                                       + r_prime_psi_ac_ts * r_prime_phi_ch_ts * shape_phi_ch_theta[j])
                                      * latent_heat * (1. - temperature_ts/melting_t)) * DimensionlessGroups::G
                                   + r_prime_psi_ac_ts * r_phi_ch_ts * latent_heat/melting_t * shape_temperature_theta[j] * DimensionlessGroups::G
                                   + (shape_c_partial_psi_theta[j] * w35_reuse_term1
                                      + c_partial_psi_ac_ts * shape_temperature_theta[j] * log_t_ts_tm) * DimensionlessGroups::Pi_T) * shape_mu_psi_ac[i];
                          //  term iii
                          mat += (-(r_prime_phi_ch_ts * shape_phi_ch_theta[j] * lambda_psi * w_prime_psi_ac_ts
                                    + r_phi_ch_ts * lambda_psi * w_prime_prime_psi_ac_ts * shape_psi_ac_theta[j])
                                  -  (shape_pressure[j] * inv_rho_partial_psi_ac_ts + pressure_star[q] * shape_inv_rho_partial_psi_theta[j]) * DimensionlessGroups::We 
                                  ) * shape_mu_psi_ac[i];
                          //  term iv
                          mat += - (r_prime_phi_ch_ts * shape_phi_ch_theta[j] * lambda_psi * grad_psi_ac_ts
                                    + (r_phi_ch_ts + artificial_diffusion_coefficient) * lambda_psi * grad_shape_psi_ac_theta[j]) * grad_shape_mu_psi_ac[i];
                              // - lambda_psi * shape_rho_theta[j] * r_phi_ch_ts * (grad_psi_ac_ts * grad_inv_rho_ts) * shape_mu_psi_ac[i]
                              // - lambda_psi * rho_ts * r_prime_phi_ch_ts * shape_phi_ch_theta[j] * (grad_psi_ac_ts * grad_inv_rho_ts) * shape_mu_psi_ac[i]
                              // - lambda_psi * rho_ts * r_phi_ch_ts * (grad_shape_psi_ac_theta[j] * grad_inv_rho_ts) * shape_mu_psi_ac[i]
                              // - lambda_psi * rho_ts * r_phi_ch_ts * (grad_psi_ac_ts * grad_shape_inv_rho_theta[j]) * shape_mu_psi_ac[i];

                          // w6
                          mat += (shape_rho_theta[j] * D_vel_Dt_ts
                                  + rho_ts * (shape_vel[j]/present_timestep + grad_shape_vel[j] * vel_bar * theta
                                              + 0.5 * div_vel_bar * shape_vel_theta[j])) * shape_vel[i]
                              -  shape_pressure[j] * shape_div_vel[i]
                              +  scalar_product(shape_eta_theta[j] * e_ts + eta_ts * shape_e_theta[j], grad_shape_vel[i]) * DimensionlessGroups::one_over_Re
                              -  scalar_product(shape_rho_theta[j] * gamma_ts + rho_ts * shape_gamma_theta[j], grad_shape_vel[i]) * DimensionlessGroups::one_over_We;

                          // w7
                          mat += (- theta * shape_div_vel[j]
                                  + (shape_inv_rho_partial_phi_theta[j] * h_ts + inv_rho_partial_phi_ch_ts * shape_h_ts[j])
                                  - (shape_inv_rho_partial_psi_theta[j] * mobility_psi * mu_psi_ac_star[q]
                                     + inv_rho_partial_psi_ac_ts * mobility_psi * shape_mu_psi_ac[j])) * shape_pressure[i];

                          // w8
                          //  term i
                          mat += ((shape_rho_theta[j] * c_ts + rho_ts * shape_c_theta[j]) * D_temperature_Dt_ts
//                                  +  rho_ts * c_ts * (shape_temperature[j]/present_timestep + shape_vel_theta[j] * grad_temperature_ts
//                                                      + vel_ts * grad_shape_temperature_theta[j])) * shape_temperature[i];
                                  +  rho_ts * c_ts * (shape_temperature[j]/present_timestep  
                                                      + vel_bar * grad_shape_temperature_theta[j])) * shape_temperature[i];
                          //  term ii
                          mat +=(-2. * ( mobility_phi * (grad_mu_phi_ch_star[q] * grad_shape_mu_phi_ch[j])
                                         + mobility_psi * mu_psi_ac_star[q] * shape_mu_psi_ac[j]) * DimensionlessGroups::one_over_Pi_T
                                 -  shape_eta_theta[j] * scalar_product(e_ts, grad_vel_ts) * DimensionlessGroups::one_over_Pi_eta
                                 -  eta_ts * scalar_product(shape_e_theta[j], grad_vel_ts) * DimensionlessGroups::one_over_Pi_eta
                                 -  eta_ts * scalar_product(e_ts, grad_shape_vel[j]) * theta * DimensionlessGroups::one_over_Pi_eta) * shape_temperature[i];
                          //  term iii
                          mat += ( k_ts * grad_shape_temperature_theta[j]
                                   + (k_partial_phi_ch_ts * shape_phi_ch_theta[j] 
                                    + k_partial_psi_ac_ts * shape_psi_ac_theta[j]) * grad_temperature_ts )
                                  * grad_shape_temperature[i] * DimensionlessGroups::one_over_Pe;
                          //  term iv
                          mat += latent_heat/melting_t * (r_prime_psi_ac_ts * shape_psi_ac_theta[j] * r_prime_phi_ch_ts * h_ts
                                                          + r_psi_ac_ts * r_prime_prime_phi_ch_ts * shape_phi_ch_theta[j] * h_ts
                                                          + r_psi_ac_ts * r_prime_phi_ch_ts * shape_h_ts[j]
                                                          - r_prime_prime_psi_ac_ts * shape_psi_ac_theta[j] * r_phi_ch_ts * mobility_psi * mu_psi_ac_star[q]
                                                          - r_prime_psi_ac_ts * r_prime_phi_ch_ts * shape_phi_ch_theta[j] * mobility_psi * mu_psi_ac_star[q]
                                                          - r_prime_psi_ac_ts * r_phi_ch_ts * mobility_psi * shape_mu_psi_ac[j])
                              * temperature_ts * shape_temperature[i] * DimensionlessGroups::one_over_Ste
                              + latent_heat/melting_t * (r_psi_ac_ts * r_prime_phi_ch_ts * h_ts
                                                         - r_prime_psi_ac_ts * r_phi_ch_ts * mobility_psi * mu_psi_ac_star[q])
                              * shape_temperature_theta[j] * shape_temperature[i] * DimensionlessGroups::one_over_Ste;
                          //  term v
                          mat += (shape_c_partial_phi_theta[j] * h_ts + c_partial_phi_ch_ts * shape_h_ts[j]
                                  - shape_c_partial_psi_theta[j] * mobility_psi * mu_psi_ac_star[q]
                                  - c_partial_psi_ac_ts * mobility_psi * shape_mu_psi_ac[j])
                              *  temperature_ts * log_t_ts_tm * shape_temperature[i]
                              +  (c_partial_phi_ch_ts * h_ts - c_partial_psi_ac_ts * mobility_psi * mu_psi_ac_star[q])
                              *  (log_t_ts_tm + 1.) * shape_temperature_theta[j] * shape_temperature[i];
#ifdef USE_AXISYMMETRY        
                          // axi-2
                          mat += + 0.5 * vel_bar[0] * one_over_r * ((shape_rho_theta[j] * vel_ts 
                                                                      + rho_ts * shape_vel_theta[j]) * shape_vel[i])
                                 - (shape_pressure[j] * shape_vel[i][0] * one_over_r)
                                 + ((shape_eta_theta[j]*(4./3.*vel_ts[0]*one_over_r-2./3.*div_vel_ts)
                                               +eta_ts*(4./3.*shape_vel_theta[j][0]*one_over_r-2./3.*theta*shape_div_vel[j]))
                                               * shape_vel[i][0] * one_over_r
                                   -2./3. * one_over_r * (shape_eta_theta[j] * vel_ts[0] + eta_ts * shape_vel_theta[j][0])
                                               * shape_div_vel[i])*DimensionlessGroups::one_over_Re;

                          // axi-4
                          mat += - shape_vel_theta[j][0] * one_over_r * shape_pressure[i];
                          // pressure reformulated term
                          mat += (shape_rho_theta[j] * pressure_reformulated_term + rho_ts * shape_pressure_reformulated_term[j]) * shape_vel[i][0] 
                               * one_over_r*DimensionlessGroups::one_over_We;
#endif
                          cell_matrix(i,j) += mat * jxwq;

                        }
                    }

                  // w1,2
                  rhs = - h_ts * shape_phi_ch[i] - mobility_phi * (grad_mu_phi_ch_star[q] * grad_shape_phi_ch[i]);

                  // w3
                  //  term i
                  rhs += - mu_phi_ch_star[q] * shape_mu_phi_ch[i];
                  // //  term ii
                  rhs += (r_psi_ac_ts * r_prime_phi_ch_ts * latent_heat * (1. - temperature_ts/melting_t) * DimensionlessGroups::G_ch
                          - c_partial_phi_ch_ts * w35_reuse_term1 * DimensionlessGroups::Pi_T_ch) * shape_mu_phi_ch[i];
                  //  term iii
                  rhs += (lambda_phi * w_prime_phi_ch_ts + lambda_psi * r_prime_phi_ch_ts * w3_reuse_term2
                          + pressure_star[q] * inv_rho_partial_phi_ch_ts * DimensionlessGroups::We) * shape_mu_phi_ch[i];
                  //  term iv
                  rhs += lambda_phi * (grad_phi_ch_ts * grad_shape_mu_phi_ch[i]);
                      // +  lambda_phi * rho_ts * (grad_phi_ch_ts * grad_inv_rho_ts) * shape_mu_phi_ch[i];

                  // w4
                  rhs +=  (- rho_ts *  D_psi_D_t_ts - mobility_psi * mu_psi_ac_star[q]) * shape_psi_ac[i];

                  // w5
                  //  term i
                  rhs += - mu_psi_ac_star[q] * shape_mu_psi_ac[i];
                  //  term ii
                  rhs += (r_prime_psi_ac_ts * r_phi_ch_ts * latent_heat * (1. - temperature_ts/melting_t) * DimensionlessGroups::G
                          - c_partial_psi_ac_ts * w35_reuse_term1 * DimensionlessGroups::Pi_T) * shape_mu_psi_ac[i];
                  //  term iii
                  rhs += (r_phi_ch_ts * lambda_psi * w_prime_psi_ac_ts + DimensionlessGroups::We * pressure_star[q] * inv_rho_partial_psi_ac_ts
                         ) * shape_mu_psi_ac[i];
                  //  term iv
                  rhs += lambda_psi * (r_phi_ch_ts + artificial_diffusion_coefficient) * (grad_psi_ac_ts * grad_shape_mu_psi_ac[i]);
                      // +  lambda_psi * rho_ts * r_phi_ch_ts * (grad_psi_ac_ts * grad_inv_rho_ts) * shape_mu_psi_ac[i];
                  if(!relax_phase_field)
                  {
                    // w6
                    rhs += - rho_ts * D_vel_Dt_ts * shape_vel[i]
                        +    pressure_star[q] * shape_div_vel[i]
                        -    eta_ts * scalar_product(e_ts, grad_shape_vel[i]) * DimensionlessGroups::one_over_Re
                        +    rho_ts * scalar_product(gamma_ts, grad_shape_vel[i]) * DimensionlessGroups::one_over_We;

                    // w7
                    rhs += (div_vel_ts - inv_rho_partial_phi_ch_ts * h_ts
                            +  inv_rho_partial_psi_ac_ts * mobility_psi * mu_psi_ac_star[q]) * shape_pressure[i];

                    // w8
                    //  term i
                    rhs += - rho_ts * c_ts * D_temperature_Dt_ts * shape_temperature[i];
                    //  term ii
                    rhs += (mobility_phi * (grad_mu_phi_ch_star[q] * grad_mu_phi_ch_star[q]) * DimensionlessGroups::one_over_Pi_T
                            + mobility_psi * (mu_psi_ac_star[q] * mu_psi_ac_star[q]) * DimensionlessGroups::one_over_Pi_T
                            + eta_ts * scalar_product(e_ts, grad_vel_ts) * DimensionlessGroups::one_over_Pi_eta) * shape_temperature[i];
                    //  term iii
                    rhs += - k_ts * grad_temperature_ts * grad_shape_temperature[i] * DimensionlessGroups::one_over_Pe;
                    //  term iv
                    rhs += - (latent_heat * (r_psi_ac_ts * r_prime_phi_ch_ts * h_ts
                              - r_prime_psi_ac_ts * r_phi_ch_ts * mobility_psi * mu_psi_ac_star[q]))
                        * temperature_ts/melting_t * shape_temperature[i] * DimensionlessGroups::one_over_Ste;
                    // term v
                    rhs += - (c_partial_phi_ch_ts * h_ts - c_partial_psi_ac_ts * mobility_psi * mu_psi_ac_star[q])
                        * temperature_ts * log_t_ts_tm * shape_temperature[i];
                  }
#ifdef USE_AXISYMMETRY
                  if(!relax_phase_field)
                  {
                    // axi-1
                    rhs += - (0.5 * rho_ts * vel_bar[0] * one_over_r * (vel_ts * shape_vel[i]))
                          + (pressure_star[q] * shape_vel[i][0] * one_over_r)
                          - (eta_ts*(4./3.*vel_ts[0]*one_over_r-2./3.*div_vel_ts) * shape_vel[i][0] * one_over_r
                                   -2./3. * eta_ts * vel_ts[0]*one_over_r* shape_div_vel[i])*DimensionlessGroups::one_over_Re;                      

                    // axi-3
                    rhs += vel_ts[0] * one_over_r * shape_pressure[i];

                    // reformulated pressure term
                    rhs += -rho_ts * pressure_reformulated_term * shape_vel[i][0] * one_over_r* DimensionlessGroups::one_over_We;
                  }
                  
#endif

                  // if(cell->index() == 17)
                  // {
                  //  pcout <<" i: " << i << " q: " << q << " rhs: " << rhs << std::endl;
                  // }
                  cell_rhs(i) += rhs * jxwq;
                }
              } // q loop on a cell

            // face loop, assemble moving contact line bc
            for (const auto face_no : cell->face_indices())
              {
                if (cell->face(face_no)->at_boundary() &&
                    cell->face(face_no)->boundary_id() == wall_boundary_id)
                // if(false)
                  {
                    fe_face_values.reinit(cell, face_no);
#ifdef USE_AXISYMMETRY
                    const auto &face_qpoints = fe_face_values.get_quadrature_points();
#endif                   
                    std::vector<double> face_phi_ch_star(n_q_points);
                    fe_face_values[extractors.phi_ch].get_function_values(current_solution, face_phi_ch_star);
                    std::vector<double> face_phi_ch_n(n_q_points);
                    fe_face_values[extractors.phi_ch].get_function_values(old_solution, face_phi_ch_n);

                    std::vector<double> face_psi_ac_star(n_q_points);
                    fe_face_values[extractors.psi_ac].get_function_values(current_solution, face_psi_ac_star);
                    std::vector<double> face_psi_ac_n(n_q_points);
                    fe_face_values[extractors.psi_ac].get_function_values(old_solution, face_psi_ac_n);

                    std::vector<Tensor<1,dim> > face_grad_phi_ch_star(n_q_points);
                    fe_face_values[extractors.phi_ch].get_function_gradients(current_solution, face_grad_phi_ch_star);
                    std::vector<Tensor<1,dim> > face_grad_phi_ch_n(n_q_points);
                    fe_face_values[extractors.phi_ch].get_function_gradients(old_solution, face_grad_phi_ch_n);

                    
                    for (unsigned int q = 0; q < n_face_q_points; ++q)
                    {
                      double face_jxwq = fe_face_values.JxW(q);
#ifdef USE_AXISYMMETRY
                      face_jxwq *= face_qpoints[q][0];
#endif
                      
                      // ts: theta star
                      const double c1 = 1. - theta;

                      const double face_phi_ch_ts =
                        theta * face_phi_ch_star[q] + c1 * face_phi_ch_n[q];

                      const double face_psi_ac_ts =
                        theta * face_psi_ac_star[q] + c1 * face_psi_ac_n[q];

                      const Tensor<1,dim> face_grad_phi_ch_ts =
                        theta * face_grad_phi_ch_star[q] + c1 * face_grad_phi_ch_n[q];

                      // const double face_inv_rho_ts =
                      //   InlineFunctions::f(face_phi_ch_ts,
                      //                      face_psi_ac_ts,
                      //                      inv_density_l,
                      //                      inv_density_s,
                      //                      inv_density_g);

                      const double face_inv_rho_partial_phi_ch_ts =
                        InlineFunctions::f_partial_phi(face_phi_ch_ts,
                                                       face_psi_ac_ts,
                                                       inv_density_l,
                                                       inv_density_s,
                                                       inv_density_g);

                      const double face_inv_rho_partial_psi_ac_ts =
                        InlineFunctions::f_partial_psi(face_phi_ch_ts,
                                                       face_psi_ac_ts,
                                                       inv_density_l,
                                                       inv_density_s);

                      // const double q_prime_phi_ch_ts = InlineFunctions::q_prime(face_phi_ch_ts);
                      // const double q_prime_prime_phi_ch_ts = InlineFunctions::q_prime_prime(face_phi_ch_ts);

                      std::vector<double> face_shape_phi_ch(dofs_per_cell);
                      std::vector<double> face_shape_psi_ac(dofs_per_cell);
                      std::vector<double> face_shape_phi_ch_theta(dofs_per_cell);
                      std::vector<double> face_shape_psi_ac_theta(dofs_per_cell);
                      std::vector<double> face_shape_inv_rho_theta(dofs_per_cell);
                      std::vector<double> face_shape_mu_phi_ch(dofs_per_cell);
                      std::vector<Tensor<1,dim> > face_grad_shape_phi_ch(dofs_per_cell);
                      std::vector<Tensor<1,dim> > face_grad_shape_phi_ch_theta(dofs_per_cell);

                      for (unsigned int k = 0; k < dofs_per_cell; ++k)
                        {
                          face_shape_phi_ch[k] = fe_face_values[extractors.phi_ch].value(k, q);
                          face_shape_psi_ac[k] = fe_face_values[extractors.psi_ac].value(k, q);

                          face_shape_phi_ch_theta[k] = face_shape_phi_ch[k] * theta;
                          face_shape_psi_ac_theta[k] = face_shape_psi_ac[k] * theta;

                          face_shape_inv_rho_theta[k] =
                            face_inv_rho_partial_phi_ch_ts * face_shape_phi_ch_theta[k]
                            + face_inv_rho_partial_psi_ac_ts * face_shape_psi_ac_theta[k];

                          face_grad_shape_phi_ch[k] = fe_face_values[extractors.phi_ch].gradient(k, q);
                          face_grad_shape_phi_ch_theta[k] = face_grad_shape_phi_ch[k] * theta;

                          face_shape_mu_phi_ch[k] = fe_face_values[extractors.mu_phi_ch].value(k, q);
                        }
                      
                      // right now only pinned, change later!
                      const bool pin_contact_line = true;

                      const auto normal_q = fe_face_values.normal_vector(q);

                      for (unsigned int i = 0; i < dofs_per_cell; ++i)
                        {
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              double wall_A;
                              if(pin_contact_line)
                                wall_A = lambda_phi * (normal_q * face_grad_shape_phi_ch_theta[j]);
                              else
                                wall_A = numbers::SQRT2 * lambda_phi * std::cos(static_contact_angle) / eps
                                                      * (1. - 2. * face_phi_ch_ts) * face_shape_phi_ch_theta[j];

                              // w3 term v
                              cell_matrix(i, j) += (wall_A
                                                    - one_over_wall_relaxation_gamma * (face_shape_phi_ch[j]/present_timestep 
                                                                                           + wall_velocity * face_grad_shape_phi_ch_theta[j]))
                                                  * face_shape_mu_phi_ch[i] * face_jxwq;
                                                    
                            }

                          double wall_A_star;
                          if(pin_contact_line)
                            wall_A_star = lambda_phi * (normal_q * face_grad_phi_ch_ts);
                          else
                            wall_A_star = numbers::SQRT2 * lambda_phi * std::cos(static_contact_angle) / eps
                                             * face_phi_ch_ts * (1. - face_phi_ch_ts);
                          // w3 term v
                          cell_rhs(i) += ( 
                                           - wall_A_star
                                           + one_over_wall_relaxation_gamma * ((face_phi_ch_star[q] - face_phi_ch_n[q])/present_timestep
                                             + wall_velocity * face_grad_phi_ch_ts)
                                          ) 
                                        * face_shape_mu_phi_ch[i] * face_jxwq;
                        }
                    }
                  }
              }            

            // stress boundary condition
            if (test_case == TestCase::test2)
              {
                for (const auto face_no : cell->face_indices())
                  {
                    if (cell->face(face_no)->at_boundary() &&
                        cell->face(face_no)->boundary_id() == 1)
                      {
                        fe_face_values.reinit(cell, face_no);
                        for (unsigned int q = 0; q < n_face_q_points; ++q)
                          for (unsigned int i = 0; i < dofs_per_cell; ++i)
                            {
                              cell_rhs(i) +=
                                -fe_face_values[extractors.velocities].value(
                                  i, q) *
                                fe_face_values.normal_vector(q) *
                                ambient_pressure * fe_face_values.JxW(q);
                            }
                      }
                  }
              } // test case
# if 1
            {
              const auto is_not_selected_component =
                [&](const unsigned int comp) {
                  if(relax_phase_field)
                  return !(
                    comp == extractors.psi_ac.component 
                    || comp == extractors.mu_psi_ac.component 
                    // || comp == extractors.pressure.component 
                    // || (comp >= extractors.velocities.first_vector_component 
                    //     && comp < extractors.velocities.first_vector_component + dim) 
                    // || comp == extractors.temperature.component
                    || comp == extractors.phi_ch.component
                    || comp == extractors.mu_phi_ch.component
                    );
                    else
                    return !(
                    comp == extractors.psi_ac.component 
                    || comp == extractors.mu_psi_ac.component 
                    || comp == extractors.pressure.component 
                    || (comp >= extractors.velocities.first_vector_component 
                        && comp < extractors.velocities.first_vector_component + dim) 
                    || comp == extractors.temperature.component
                    || comp == extractors.phi_ch.component
                    || comp == extractors.mu_phi_ch.component
                    );
                };
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  const unsigned int component_i =
                    fe.system_to_component_index(i).first;
                  if (is_not_selected_component(component_i))
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          cell_matrix(i, j) = (i == j) ? 1. : 0.;
                        }
                      cell_rhs(i) = 0;
                    }
                  else
                    {
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          const unsigned int component_j =
                            fe.system_to_component_index(j).first;
                          if (is_not_selected_component(component_j))
                            {
                              cell_matrix(i, j) = 0.;
                            }
                        }
                    }
                }
            }
# endif            
            // pcout<<" cell index: "<<cell->index()<<std::endl;
            constraints_newton_update.distribute_local_to_global(cell_matrix,
                                                                 cell_rhs,
                                                                 local_dof_indices,
                                                                 system_matrix,
                                                                 system_rhs);
          }// cell loop

    system_matrix.compress(VectorOperation::add);
//    preconditioner_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    timer.stop();
    pcout<<" assemble time: "<<timer.wall_time()<<std::endl;    
}

template <int dim>
void StokesProblem<dim>::newton_iteration()
{
  TimerOutput::Scope t(computing_timer, "Newton");
  pcout<< "Newton iteration" << std::endl;

  // set to 1 for testing
  const unsigned int max_iter = 20;
  bool assemble_matrix = false;
  assemble_system(assemble_matrix);
  const double initial_residual = system_rhs.l2_norm();
  double residual = initial_residual;
  double residual_old = initial_residual + 1.;
  double alpha = 1.;

  pcout << "initial residual=" << residual << std::endl;
// #define USE_BLOCKDIRECT_SOLVER

#ifdef USE_PETSC_LA
  SolverControl cn;
  PETScWrappers::SparseDirectMUMPS solver(cn, mpi_communicator);
#else
  SolverControl                  solver_control(2000, 1e-8);

  TrilinosWrappers::SolverDirect::AdditionalData data;
  data.solver_type = "Amesos_Umfpack";
  
  // TrilinosWrappers::SolverDirect solver(solver_control, data);

  TrilinosWrappers::PreconditionBlockwiseDirect preconditioner;

  // SolverGMRES<VectorType> solver(solver_control);
  SolverBicgstab<VectorType> solver(solver_control);

  // TrilinosWrappers::SolverDirect::AdditionalData data;
  // TrilinosWrappers::SolverDirect                 solver(solver_control, data);
#endif

  assemble_matrix = true;
  VectorType locally_owned_solution(system_rhs);
  VectorType locally_owned_solution_tmp(system_rhs);
  locally_owned_solution = current_solution;
  for (unsigned int k = 1; k <= max_iter; ++k) {
      if (residual < 1.e-6 )
        break;
      assemble_system(assemble_matrix);
      {
        TimerOutput::Scope t(computing_timer, "Direct Solve");
        // try
        //   {
        //     Timer timer(mpi_communicator);
        //     preconditioner.initialize(system_matrix);
        //     solver.solve(system_matrix,
        //                  newton_update,
        //                  system_rhs,
        //                  preconditioner);

        //     pcout << " cg number of iterations: " << solver_control.last_step()
        //           << " cg solve time: "<< timer.wall_time()
        //           << std::endl;
        //   }
        // catch (const std::exception &exc)
          {
            TimerOutput::Scope t(computing_timer, "Direct Solve trillinos");
            Timer timer(mpi_communicator);
            pcout << " cg failed, use direct solver: " << std::endl;
            TrilinosWrappers::SolverDirect::AdditionalData data;
            TrilinosWrappers::SolverDirect solver(solver_control, data);
            solver.solve(system_matrix, newton_update, system_rhs);
            timer.stop();
            pcout<<" direct solver time: "<<timer.wall_time()<<std::endl;
          }
        }


      pcout<<" mat norm: "<<system_matrix.frobenius_norm()
           <<" rh2 norm: "<< system_rhs.l2_norm()
           <<" sol norm: "<< newton_update.l2_norm()<<std::endl;

      constraints_newton_update.distribute(newton_update);

      // locally_owned_solution += newton_update;
      alpha = 1.;
      locally_owned_solution.add(alpha, newton_update);
      current_solution = locally_owned_solution;

      // {
      //   locally_relevant_solution = current_solution;
      //   output_results(1000000+k);
      // }
      
      residual_old = residual;
      assemble_system(false);
      residual = system_rhs.l2_norm();
      pcout << "k= " << k << "  residual = " << residual <<  std::endl;

      if(residual > residual_old)
        {
          // backtracking is trigged
          pcout<<" backtracking is trigged "
          <<" residual = "<<residual<<" residual_old = "<<residual_old<<std::endl;
          double alpha3 =1.;
          double g3 = residual;
          const double g1 = residual_old;

          for(unsigned int m=0; m<30; ++m)
          {
            alpha3 *= 0.5;
            locally_owned_solution_tmp = locally_owned_solution;
            locally_owned_solution_tmp.add(alpha3, newton_update);
            current_solution = locally_owned_solution_tmp;
            assemble_system(false);
            g3 = system_rhs.l2_norm();
            pcout<<" m = " << m << " g3 = "<<g3<<" alpha3 = "<<alpha3<<std::endl;
            if(g3 < residual_old)
              break;
          }

          const double alpha2 = 0.5 * alpha3;
          locally_owned_solution_tmp = locally_owned_solution;
          locally_owned_solution_tmp.add(alpha2, newton_update);
          current_solution = locally_owned_solution_tmp;
          assemble_system(false);
          const double g2 = system_rhs.l2_norm();
          pcout<<" g2 = "<<g2<< " alpha2 = " << alpha2 <<std::endl;

          if(g2<g3)
          {
            alpha = alpha2;
            residual = g2;
          }
          else
          {
            alpha = alpha3;
            residual = g3;
          }

          // find alpha0
          const double h1 = (g2-g1)/alpha2;
          const double h2 = (g3-g2)/(alpha3-alpha2);
          const double h3 = (h2-h1)/alpha3;
          const double alpha0 = (alpha2 - h1/h3) * 0.5;
          locally_owned_solution_tmp = locally_owned_solution;
          locally_owned_solution_tmp.add(alpha0, newton_update);
          current_solution = locally_owned_solution_tmp;
          assemble_system(false);
          const double g0 = system_rhs.l2_norm();
          pcout<<" g0 = "<<g0 <<" alpha0 = "<< alpha0 <<std::endl;

          if(residual > g0)         
          {
            alpha = alpha0;
            residual = g0;
          }

          pcout<<" residual = "<<residual<<" alpha = "<<alpha<<std::endl;
                // locally_owned_solution += newton_update;
          locally_owned_solution.add(alpha, newton_update);
          current_solution = locally_owned_solution;
        }


    }
  locally_relevant_solution = current_solution;

}

template <int dim>
void
StokesProblem<dim>::map_dofs_to_component(
  const DoFHandler<dim> &    dof,
  std::vector<unsigned int> &global_index_to_component)
{
  global_index_to_component.resize(dof_handler.n_dofs(), numbers::invalid_unsigned_int);
  std::vector<types::global_dof_index> local_dof_indices;
  // store the components of each global index.
  for (const auto &cell : dof.active_cell_iterators())
    if(cell->is_locally_owned())
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
  pcout << "   Refine..." << std::endl;

  const bool refine_mesh_local =
    label_mesh<dim, VectorType>(dof_handler,
                                fe,
                                locally_relevant_solution,
                                min_mesh_size,
                                max_mesh_size,
                                0,
                                n_refinement,
                                extractors);

  const bool refine_mesh = Utilities::MPI::logical_or(refine_mesh_local,
                             mpi_communicator);

  if(refine_mesh)
  {
    pcout<<"Refine mesh..."<<std::endl;
    parallel::distributed::SolutionTransfer<dim, VectorType> solution_trans(
    dof_handler);

    std::vector<const VectorType *> in_solution(2);
    in_solution[0] = &locally_relevant_solution;
    in_solution[1] = &old_solution;

    triangulation.prepare_coarsening_and_refinement();

    solution_trans.prepare_for_coarsening_and_refinement(in_solution);

    triangulation.execute_coarsening_and_refinement();

#ifdef USE_DIRECT_SOLVER
    setup_system();
#else
    setup_block_system();
#endif    
    
    VectorType distributed_sol1(system_rhs);
    VectorType distributed_sol2(system_rhs);
    std::vector<VectorType *> tmp(2);
    tmp[0] = &distributed_sol1;
    tmp[1] = &distributed_sol2;
    solution_trans.interpolate(tmp);

    constraints_boundary.distribute(distributed_sol1);
    constraints_boundary.distribute(distributed_sol2);

    locally_relevant_solution = distributed_sol1;
    old_solution = distributed_sol2;
  }
  else
  {
    pcout << "No refinement" << std::endl;
  }

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

    data_out.build_patches(2);

    // have to create the directory output
    data_out.write_vtu_with_pvtu_record(
        output_dir, "solution", cycle, mpi_communicator, 5, 1);
}

template <int dim>
void StokesProblem<dim>::print_variables() const 
{
  pcout << " eps:                  " << eps << std::endl
        << " density_s:            " << density_s << std::endl
        << " density_l:            " << density_l << std::endl
        << " density_g:            " << density_g << std::endl
        << " cl:                   " << cl << std::endl
        << " cs:                   " << cs << std::endl
        << " cg:                   " << cg << std::endl
        << " eta_l:                " << eta_l << std::endl
        << " eta_s:                " << eta_s << std::endl
        << " eta_g:                " << eta_g << std::endl
        << " surface_tension_ac:   " << surface_tension_psi_ac << std::endl
        << " surface_tension_ch:   " << surface_tension_phi_ch << std::endl
        << " lambda_phi:           " << lambda_phi << std::endl
        << " lambda_psi:           " << lambda_psi << std::endl
        << " mobility_phi:         " << mobility_phi << std::endl
        << " mobility_psi:         " << mobility_psi << std::endl
        << " latent_heat:          " << latent_heat << std::endl
        << " melting_t:            " << melting_t << std::endl
        << " initial_temperature:  " << initial_temperature << std::endl
        << " k_l:                  " << k_l << std::endl
        << " k_s:                  " << k_s << std::endl
        << " k_g:                  " << k_g << std::endl
        << " ambient_pressure:     " << ambient_pressure << std::endl
        << " num of MPI processes: " << Utilities::MPI::n_mpi_processes(mpi_communicator) << std::endl;
#ifdef USE_AXISYMMETRY
  pcout << " axisymetric:          " << "true" << std::endl;
#else
  pcout << " axisymetric:          " << "false" << std::endl;  
#endif    
#ifdef USE_NEW_R
  pcout << " new_r:                " << "true" << std::endl;
#else
  pcout << " new_r:                " << "false" << std::endl;   
#endif

}

template <int dim>
void
StokesProblem<dim>::test_adaptive_refinement()
{
  const bool refine_mesh =
    label_mesh<dim, VectorType>(dof_handler,
                                fe,
                                locally_relevant_solution,
                                min_mesh_size,
                                max_mesh_size,
                                0,
                                n_refinement,
                                extractors);
  if(refine_mesh)                                
  {
    pcout << "Refine mesh\n" << std::flush;
    parallel::distributed::SolutionTransfer<dim, VectorType> solution_trans(
      dof_handler);

    std::vector<const VectorType *> in_solution(2);
    in_solution[0] = &locally_relevant_solution;
    in_solution[1] = &old_solution;

    triangulation.prepare_coarsening_and_refinement();

    solution_trans.prepare_for_coarsening_and_refinement(in_solution);

    triangulation.execute_coarsening_and_refinement();

#ifdef USE_DIRECT_SOLVER
    setup_system();
#else
    setup_block_system();
#endif

    VectorType                distributed_sol1(system_rhs);
    VectorType                distributed_sol2(system_rhs);
    std::vector<VectorType *> tmp(2);
    tmp[0] = &distributed_sol1;
    tmp[1] = &distributed_sol2;
    solution_trans.interpolate(tmp);

    constraints_boundary.distribute(distributed_sol1);
    constraints_boundary.distribute(distributed_sol2);

    locally_relevant_solution = distributed_sol1;
    old_solution              = distributed_sol2;
  }
}

template <int dim>
void
StokesProblem<dim>::save_checkpoint(const unsigned int step_number,
                                    const double       runtime)
{
  TimerOutput::Scope t(computing_timer, "save checkpoints");
  pcout<<" save checkpoint to \n"<< checkpoints_dir + "restart.mesh" <<std::flush;
  {
    // mesh and vectors
    std::vector<const VectorType *> solutions(3);
    solutions[0] = &locally_relevant_solution;
    solutions[1] = &old_solution;
    solutions[2] = &old_old_solution;

    for (unsigned int i = 0; i < solutions.size(); ++i)
      {
        VectorType tmp_sol(system_rhs);
        tmp_sol = *solutions[i];
        pcout << " solution l2 norm: " << tmp_sol.l2_norm() << std::endl;
      }

    parallel::distributed::SolutionTransfer<dim, VectorType> solution_trans(
      dof_handler);

    solution_trans.prepare_for_serialization(solutions);
    triangulation.save(checkpoints_dir + "restart.mesh");
  }

  {
    // other parameters
    std::ofstream ofs(checkpoints_dir + "restart.dat");
    boost::archive::binary_oarchive ar(ofs);

    ar & step_number;
    ar & runtime;
    ar & present_timestep;
  }
}

template <int dim>
void
StokesProblem<dim>::load_checkpoint(unsigned int &step_number,
                                   double       &runtime)
{
  const unsigned int step_number_old = step_number;
  std::string step_number_string;
  if(step_number != 0)
    step_number_string = Utilities::int_to_string(step_number, 5);  
  pcout<<" load checkpoint from: "
  << (checkpoints_dir + "restart-" + step_number_string + ".mesh")
  << std::flush;
  

  {
    create_coarse_grid(triangulation);
    pcout << " n_levels: " << triangulation.n_levels() << std::endl;
    
    triangulation.load(checkpoints_dir + "restart-" + step_number_string + ".mesh");
    // set_boundary_ids();
    print_mesh_info(triangulation);
    // dof_handler.distribute_dofs(fe);
    // pcout << " n_dofs: " << dof_handler.n_dofs() << std::endl;

    setup_system();

    parallel::distributed::SolutionTransfer<dim, VectorType> solution_trans(
      dof_handler);

    const unsigned int      n_vectors = 3;
    std::vector<VectorType> solutions(n_vectors);
    for (unsigned int i = 0; i < n_vectors; ++i)
      solutions[i].reinit(dof_handler.locally_owned_dofs(), mpi_communicator);

    std::vector<VectorType *> x_system(3);
    int                       i = 0;
    for (auto &v : x_system)
      {
        v = &solutions[i];
        ++i;
      }

    solution_trans.deserialize(x_system);
    for (unsigned int i = 0; i < solutions.size(); ++i)
      {
        pcout << " solution l2 norm: " << solutions[i].l2_norm() << std::endl;
      }
    locally_relevant_solution = solutions[0];
    old_solution              = solutions[1];
    old_old_solution          = solutions[2];
  }

  {
    // load other parameters:
    // std::ifstream                   ifs("checkpoint.dat");
    std::ifstream                   ifs(checkpoints_dir + "restart-" + step_number_string + ".dat");
    boost::archive::binary_iarchive ar(ifs);

    ar &step_number;
    ar &runtime;
    ar &present_timestep;
    if(step_number_old !=0)
      AssertDimension(step_number, step_number_old);

  }
  pcout<<" loaded checkpoint from step: "<<step_number<<" runtime: "<<runtime<<std::endl;
}

template <int dim>
void
StokesProblem<dim>::run()
{
  const std::string prefix = "tmp476/";

  output_dir      = "./output/" + prefix;
  checkpoints_dir = "./checkpoints/" + prefix;
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      const auto tmp  = system(("mkdir " + output_dir).c_str());
      const auto tmp1 = system(("mkdir " + checkpoints_dir).c_str());
      (void)tmp;
      (void)tmp1;
    }
#ifdef USE_PETSC_LA
    pcout << "Running using PETSc." << std::endl;
#else
    pcout << "Running using Trilinos." << std::endl;
#endif
    pcout << "n refinement " << n_refinement << ':' << std::endl;
    pcout << " output_dir: " << output_dir << std::endl;

    step_number = 0;
    runtime           = 0.;

    const unsigned int max_step_number =1500;
    const unsigned int output_interval = 5;
    const unsigned int checkpoint_output_interval = 100;
    const unsigned int save_checkpoint_interval = 10; // smaller than or equal to checkpoint_output_interval

    const bool start_from_checkpoint = false;

    if(!start_from_checkpoint)
      {
        make_grid();

#ifdef USE_DIRECT_SOLVER
        setup_system();
#else
        setup_block_system();
#endif

        setup_initial_condition();
        relax_phase_field = true;
        const unsigned int n_relaxation_steps = 0;
        // output_results(0);
        if(relax_phase_field)
          for(unsigned int i=0; i<n_relaxation_steps; ++i)
            {
              pcout<<" relaxation phase field "<<i<<std::endl;
              present_timestep = 1e-6;
              // if(i==0)
              //   theta = 1.;
              // else 
              //   theta = 0.5;
              newton_iteration();
              old_old_solution = old_solution;              // n-1
              old_solution     = locally_relevant_solution; // n
              current_solution = old_solution; // u^*, newton initial guess
              // output_results(i+1);
              pcout<<std::endl;
            }
        pcout<<" done relaxation phase field "<<std::endl;            
        relax_phase_field = false;
        // exit(0);



        // {
        //   // relaxation of psi and phi
        //   VectorType tmp_initial_sol;

        //   tmp_initial_sol.reinit(dof_handler.locally_owned_dofs(),
        //                          mpi_communicator);
        //   tmp_initial_sol = locally_relevant_solution;
        //   ComponentMask temperature_mask(fe.n_components(), false);
        //   temperature_mask.set(extractors.temperature.component, true);
        //   VectorTools::interpolate(
        //     dof_handler,
        //     Functions::ConstantFunction<dim>(melting_t, fe.n_components()),
        //     tmp_initial_sol,
        //     temperature_mask);
        //   // constraints_boundary.distribute(tmp_initial_sol);
        //   locally_relevant_solution = tmp_initial_sol;
        //   old_solution              = locally_relevant_solution;
        //   old_old_solution          = locally_relevant_solution;
        //   current_solution          = old_solution;
        //   output_results(0);
        //   newton_iteration();
        //   output_results(1);
        //   exit(0);
        // }
      }
    else
    {
      // restart step number
      step_number = 700;
      load_checkpoint(step_number, runtime);
    }

    output_results(step_number);

    // test_adaptive_refinement();
    // output_results(step_number+1);
    // exit(0);

    while (step_number < max_step_number)
      {
        old_timestep     = present_timestep;
        present_timestep = std::min(fix_timestep, (1e-5) * std::pow(1.05, step_number));        

        step_number ++;
        runtime += present_timestep;
        // if (step_number == 1)
        //   theta = 1.;
        // else
        //   theta = 0.5;
        pcout << "###starting step " << step_number << "  dt= " << present_timestep
              << "  time= " << runtime 
              << " theta= " << theta
              << std::endl;

        if(step_number > 1 && use_adaptive_refinement)
          refine_grid();

        old_old_solution = old_solution;          // n-1
        old_solution = locally_relevant_solution; // n
        current_solution = old_solution; // u^*, newton initial guess

        newton_iteration();

        // {
        //   // limit psi
        //   std::vector<unsigned int> global_index_to_component(dof_handler.n_dofs(), numbers::invalid_unsigned_int);
        //   map_dofs_to_component(dof_handler, global_index_to_component);
        //   for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
        //     if (locally_relevant_solution.in_local_range(i))
        //       {
        //         const unsigned int component_i = global_index_to_component[i];
        //         if (component_i == extractors.psi_ac.component)
        //           {
        //             double psi_i = locally_relevant_solution(i);
        //             InlineFunctions::limit_phase_field_function(psi_i);
        //             locally_relevant_solution(i) = psi_i;
        //           }
        //       }
        // }

        if ((step_number % output_interval == 0)||(step_number <21 ))
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(step_number);
          }

      if(step_number%save_checkpoint_interval == 0)
        save_checkpoint(step_number, runtime);

      if(step_number % checkpoint_output_interval == 0)
      {
        if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        {
          const std::string filename = checkpoints_dir + "restart-" + Utilities::int_to_string(step_number, 5);
          move_file(checkpoints_dir + "restart.mesh", filename + ".mesh");
          move_file(checkpoints_dir + "restart.mesh.info", filename + ".mesh.info");
          move_file(checkpoints_dir + "restart.mesh_fixed.data", filename + ".mesh_fixed.data");
          move_file(checkpoints_dir + "restart.dat", filename + ".dat");
        }
      }

      pcout << std::endl;
      pcout << std::endl;
      }
    computing_timer.print_summary();
    computing_timer.reset();
}
} // namespace Step55



int main(int argc, char *argv[])
{
    try
    {
        using namespace dealii;
        using namespace Step55;

        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
        const TestCase testcase = TestCase::test3;
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
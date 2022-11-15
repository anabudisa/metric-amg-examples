# Collection of models leading to interface perturbed operators

## Problems in $2d$
These things are mostly done on `UnitSquareMesh` geometries to check correctness
- [x] EMI model
- [x] Reduced $2d$ - $1d$ EMI model 
- [x] Bidomain model
- [x] vector Poisson-vector Poisson coupling by full vector (arises in Elasticity-Poroelasticity)
- [x] vector Poisson-vector Poisson coupling by tangent vector components (arises in Biot-Stokes)
- [x] EMI like model with coupling over a (thin) subdomain 

## Problems in $3d$
These things are mostly done on `UnitCubeMesh` geometries to check correctness
- [x] EMI model
- [ ] Reduced $3d$ - $1d$ EMI model 
- [x] Bidomain model
- [x] vector Poisson-vector Poisson coupling by full vector (arises in Elasticity-Poroelasticity)
- [x] vector Poisson-vector Poisson coupling by tangent vector components (arises in Biot-Stokes)
- [x] EMI like model with coupling over a (thin) subdomain

## Others
- [x] Reduced $3d$ - $1d$ EMI model on cube with Sylvie Lorthois [network](https://www.nature.com/articles/s41467-021-27534-8)
- [x] Reduced $3d$ - $1d$ EMI model on brain + Brava
- [x] Biot-Stokes on brain geometry
- [ ] (**maybe**) Introduce fractal interface in Reduced $2d$-$1d$ EMI model
- [ ] (**maybe**) Staying with hypercube add more challenging interface

## Dependencies
- FEniCS stack
- FEniCS_ii
- gmshnics
- networkx
- For meshes in `./data/` please reach out for how to cite

<p align="center">
  <img src="https://github.com/MiroK/metric-amg-examples/blob/master/doc/sylvie.png">
</p>

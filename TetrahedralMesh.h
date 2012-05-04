/***************************************************************************/
/*                       CUDA based TLED Solver                            */
/*                     {c} 2008-2010 Karsten Noe                           */
/*                      The Alexandra Institute                            */
/*                   See out blog on cg.alexandra.dk                       */ 
/***************************************************************************/

#ifndef TETRAHEDRALMESH
#define TETRAHEDRALMESH

#include </usr/local/cuda/include/vector_types.h>

namespace TLED {
#include <vector_types.h>

typedef float4 Point;

typedef int4 Tetrahedron; 

typedef uint3 Triangle; 

struct TriangleSurface
{
	Triangle* faces;
	int numFaces;
};

struct TetrahedralMesh
{
	Point * points;
	int numPoints;
	Tetrahedron * tetrahedra;
	int numTetrahedra;
	int4 *writeIndices;
	int numWriteIndices;
	float *volume;
	float *mass;
};

struct ShapeFunctionDerivatives
{
	float3 h1; // derivatives at node 1 w.r.t. (x,y,z)
	float3 h2; // derivatives at node 2 w.r.t. (x,y,z)
	float3 h3; // derivatives at node 3 w.r.t. (x,y,z)
	float3 h4; // derivatives at node 4 w.r.t. (x,y,z)
};

struct TetrahedralTLEDState
{
	float4 *ABC, *Ui_t, *Ui_tminusdt, *pointForces, *externalForces;
	ShapeFunctionDerivatives *shape_function_deriv;
	int maxNumForces;
	float mu, lambda;
	float timeStep;
};

}
#endif

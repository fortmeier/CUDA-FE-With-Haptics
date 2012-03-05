/***************************************************************************/
/*                       CUDA based TLED Solver                            */
/*                     {c} 2008-2010 Karsten Noe                           */
/*                      The Alexandra Institute                            */
/*                   See our blog on cg.alexandra.dk                       */ 
/***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "TetrahedralMesh.h"


__global__ void
precalculateShapeFunctionDerivatives_k(ShapeFunctionDerivatives *shape_function_derivatives, Tetrahedron *tetrahedra, Point *points, unsigned int numTets)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numTets)
		return;

	ShapeFunctionDerivatives sfd;
	
	Tetrahedron tet = tetrahedra[me_idx];

	if (tet.x<0) return; // illegal tetrahedron

	float4 a = points[tet.x];
	float4 b = points[tet.y];
	float4 c = points[tet.z];
	float4 d = points[tet.w];

	float denominator = c.y*d.x*b.z + a.x*c.y*d.z + a.y*d.x*c.z - c.x*b.y*a.z + c.x*d.y*a.z - a.x*b.y*d.z +
		a.x*b.y*c.z + c.x*b.y*d.z - c.x*a.y*d.z + a.y*c.x*b.z - a.y*d.x*b.z - d.x*b.y*c.z + b.x*c.y*a.z + 
		b.x*d.y*c.z - b.x*a.y*c.z - b.x*c.y*d.z - b.x*d.y*a.z + b.x*a.y*d.z - d.y*c.x*b.z + d.y*a.x*b.z -
		d.y*a.x*c.z - c.y*a.x*b.z - d.x*c.y*a.z + d.x*b.y*a.z;

	sfd.h1.x = (c.y*d.z - b.y*d.z + b.y*c.z + d.y*b.z - d.y*c.z - c.y*b.z)/denominator;
	sfd.h1.y = -(-c.x*b.z + d.x*b.z + c.x*d.z - b.x*d.z + b.x*c.z - d.x*c.z)/denominator;
	sfd.h1.z = (-c.x*b.y + c.x*d.y - b.x*d.y + b.x*c.y - d.x*c.y + d.x*b.y)/denominator;

	sfd.h2.x = -(c.y*d.z - c.y*a.z + d.y*a.z - a.y*d.z + c.z*a.y - d.y*c.z)/denominator;
	sfd.h2.y = (c.x*d.z - a.x*d.z - c.x*a.z + d.x*a.z - d.x*c.z + a.x*c.z)/denominator;
	sfd.h2.z = -(-a.x*d.y + a.x*c.y + d.x*a.y - c.x*a.y + c.x*d.y - d.x*c.y)/denominator;

	sfd.h3.x = (-d.y*b.z + a.y*b.z - a.y*d.z + b.y*d.z + d.y*a.z - b.y*a.z)/denominator;
	sfd.h3.y = -(d.x*a.z - b.x*a.z - d.x*b.z + a.x*b.z - a.x*d.z + b.x*d.z)/denominator;
	sfd.h3.z = (-a.x*d.y + d.x*a.y - b.x*a.y - d.x*b.y + a.x*b.y + b.x*d.y)/denominator;

	sfd.h4.x = -(-c.z*a.y + a.y*b.z + b.y*c.z + c.y*a.z - b.y*a.z - c.y*b.z)/denominator;
	sfd.h4.y = (-a.x*c.z + c.x*a.z - b.x*a.z + b.x*c.z + a.x*b.z - c.x*b.z)/denominator;
	sfd.h4.z = -(-a.x*c.y - b.x*a.y + b.x*c.y + a.x*b.y - c.x*b.y + c.x*a.y)/denominator;

/*	printf("\nFor tetrahedron %i: \n", me_idx);
	printf("h1 derivatives: %f, %f, %f \n", sfd.h1.x, sfd.h1.y, sfd.h1.z);
	printf("h2 derivatives: %f, %f, %f \n", sfd.h2.x, sfd.h2.y, sfd.h2.z);
	printf("h3 derivatives: %f, %f, %f \n", sfd.h3.x, sfd.h3.y, sfd.h3.z);
	printf("h4 derivatives: %f, %f, %f \n", sfd.h4.x, sfd.h4.y, sfd.h4.z);
*/
	shape_function_derivatives[me_idx] = sfd;

}

__global__ void
precalculateABC(float4* ABCm, float* M, float timestep, float alpha, unsigned int numPoints)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numPoints)
		return;

	float twodelta = timestep*2.0f;
	float deltasqr = timestep*timestep;


	float Mii = M[me_idx];
	float Dii = alpha*Mii;  // mass-proportional damping is applied
	
//	printf("M: %f\n",Mii);

	float Ai = 1.0f/(Dii/twodelta + Mii/deltasqr);
	float Bi = ((2.0f*Mii)/deltasqr)*Ai;
	float Ci = (Dii/twodelta)*Ai - 0.5f*Bi;

//	printf("ABC for node %i: %f, %f, %f \n", me_idx, Ai, Bi, Ci);

	ABCm[me_idx] = make_float4(Ai,Bi,Ci,Mii);
}


__global__ void
updateDisplacements_k(float4 *Ui_t, float4 *Ui_tminusdt, float *M, float4 *Ri, float4 *Fi, int maxNumForces, float4 *ABC, unsigned int numPoints)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numPoints)
		return;

	float4 F = make_float4(0,0,0,0);

//	printf("Max num forces: %i\n", maxNumForces);

	for (int i=0; i<maxNumForces; i++)
	{
		float4 force_to_add = Fi[me_idx*maxNumForces+i];
		F.x += force_to_add.x;
		F.y += force_to_add.y;
		F.z += force_to_add.z;
	}
//	printf("Accumulated node %i force: %f, %f, %f \n", me_idx, F.x, F.y, F.z);

	float4 ABCi = ABC[me_idx];
	float4 Uit = Ui_t[me_idx];
	float4 Uitminusdt = Ui_tminusdt[me_idx];

	float4 R = Ri[me_idx];
	float x = ABCi.x * (R.x - F.x) + ABCi.y * Uit.x + ABCi.z * Uitminusdt.x;
	float y = ABCi.x * (R.y - F.y) + ABCi.y * Uit.y + ABCi.z * Uitminusdt.y;
	float z = ABCi.x * (R.z - F.z) + ABCi.y * Uit.z + ABCi.z * Uitminusdt.z;

/*	float x = ABCi.x * (-F.x) + ABCi.y * Ui_t[me_idx].x + ABCi.z * Ui_tminusdt[me_idx].x;
	float y = ABCi.x * (-F.x) + ABCi.y * Ui_t[me_idx].y + ABCi.z * Ui_tminusdt[me_idx].y;
	float z = ABCi.x * (-F.x ) + ABCi.y * Ui_t[me_idx].z + ABCi.z * Ui_tminusdt[me_idx].z;
*/
	Ui_tminusdt[me_idx] = make_float4(x,y,z,0);//XXXXXXXXXXXXXXXXXXXXX

}

struct Matrix4x3 //note: supposed to be castable to a ShapeFunctionDerivatives object
{
	float e[12];
};

struct Matrix3x3 
{
	float e[9];
};

struct Matrix6x3 
{
	float e[6*3];
};


texture<float4,  1, cudaReadModeElementType> Ui_t_1d_tex;
texture<float,  1, cudaReadModeElementType> V0_1d_tex;
texture<float4,  1, cudaReadModeElementType> _tex;

#define h(i,j) (sfdm.e[(i-1)*3+(j-1)])
#define u(i,j) (displacements.e[(i-1)*3+(j-1)])
#define X(i,j) (deformation_gradients.e[(i-1)*3+(j-1)])
#define B(i,j) (b_tensor.e[(i-1)*3+(j-1)])
#define C(i,j) (cauchy_green_deformation.e[(i-1)*3+(j-1)])
#define CI(i,j) (c_inverted.e[(i-1)*3+(j-1)])
#define S(i,j) (s_tensor.e[(i-1)*3+(j-1)])

__global__ void
calculateForces_k(Matrix4x3 *shape_function_derivatives, Tetrahedron *tetrahedra, float4 *Ui_t, float *V_0, int4 *writeIndices, float4 *pointForces, int maxPointForces, float mu, float lambda, unsigned int numTets)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numTets)
		return;

	Tetrahedron e = tetrahedra[me_idx];

	if (e.x < 0) 
		return;
	
	Matrix4x3 sfdm = shape_function_derivatives[me_idx];
	Matrix4x3 displacements;

	//fill in displacement values in u (displacements)
	
	float3 U1 = crop_last_dim(tex1Dfetch( Ui_t_1d_tex, e.x ));
	float3 U2 = crop_last_dim(tex1Dfetch( Ui_t_1d_tex, e.y ));
	float3 U3 = crop_last_dim(tex1Dfetch( Ui_t_1d_tex, e.z ));
	float3 U4 = crop_last_dim(tex1Dfetch( Ui_t_1d_tex, e.w ));

	displacements.e[0] = U1.x;
	displacements.e[1] = U1.y;
	displacements.e[2] = U1.z;

	displacements.e[3] = U2.x;
	displacements.e[4] = U2.y;
	displacements.e[5] = U2.z;

	displacements.e[6] = U3.x;
	displacements.e[7] = U3.y;
	displacements.e[8] = U3.z;

	displacements.e[9] = U4.x;
	displacements.e[10] = U4.y;
	displacements.e[11] = U4.z;

	Matrix3x3 deformation_gradients;

	//Calculate deformation gradients
	X(1,1) = (u(1,1)*h(1,1)+u(2,1)*h(2,1)+u(3,1)*h(3,1)+u(4,1)*h(4,1)+1.0f); 
	X(1,2) = (u(1,1)*h(1,2)+u(2,1)*h(2,2)+u(3,1)*h(3,2)+u(4,1)*h(4,2));
	X(1,3) = (u(1,1)*h(1,3)+u(2,1)*h(2,3)+u(3,1)*h(3,3)+u(4,1)*h(4,3));

	X(2,1) = (u(1,2)*h(1,1)+u(2,2)*h(2,1)+u(3,2)*h(3,1)+u(4,2)*h(4,1));
	X(2,2) = (u(1,2)*h(1,2)+u(2,2)*h(2,2)+u(3,2)*h(3,2)+u(4,2)*h(4,2)+1.0f);
	X(2,3) = (u(1,2)*h(1,3)+u(2,2)*h(2,3)+u(3,2)*h(3,3)+u(4,2)*h(4,3));

	X(3,1) = (u(1,3)*h(1,1)+u(2,3)*h(2,1)+u(3,3)*h(3,1)+u(4,3)*h(4,1));
	X(3,2) = (u(1,3)*h(1,2)+u(2,3)*h(2,2)+u(3,3)*h(3,2)+u(4,3)*h(4,2));
	X(3,3) = (u(1,3)*h(1,3)+u(2,3)*h(2,3)+u(3,3)*h(3,3)+u(4,3)*h(4,3)+1.0f);

/*	printf("\nDeformation gradient tensor for tetrahedron %i: \n", me_idx);
	printf("%f, %f, %f \n", X(1,1), X(1,2), X(1,3));
	printf("%f, %f, %f \n", X(2,1), X(2,2), X(2,3));
	printf("%f, %f, %f \n", X(3,1), X(3,2), X(3,3));
*/
		// calculate Right Cauchy-Green deformation tensor C
		Matrix3x3 cauchy_green_deformation;

	C(1,1) = X(1, 1)*X(1, 1) + X(2, 1)*X(2, 1) + X(3, 1)*X(3, 1); 
	C(1,2) = X(1, 1)*X(1, 2) + X(2, 1)*X(2, 2) + X(3, 1)*X(3, 2); 
	C(1,3) = X(1, 1)*X(1, 3) + X(2, 1)*X(2, 3) + X(3, 1)*X(3, 3); 

	C(2,1) = X(1, 1)*X(1, 2) + X(2, 1)*X(2, 2) + X(3, 1)*X(3, 2); 
	C(2,2) = X(1, 2)*X(1, 2) + X(2, 2)*X(2, 2) + X(3, 2)*X(3, 2); 
	C(2,3) = X(1, 2)*X(1, 3) + X(2, 2)*X(2, 3) + X(3, 2)*X(3, 3);

	C(3,1) = X(1, 1)*X(1, 3) + X(2, 1)*X(2, 3) + X(3, 1)*X(3, 3); 
	C(3,2) = X(1, 2)*X(1, 3) + X(2, 2)*X(2, 3) + X(3, 2)*X(3, 3); 
	C(3,3) = X(1, 3)*X(1, 3) + X(2, 3)*X(2, 3) + X(3, 3)*X(3, 3);
/*
	printf("\nRight Cauchy-Green deformation tensor for tetrahedron %i: \n", me_idx);
	printf("%f, %f, %f \n", C(1,1), C(1,2), C(1,3));
	printf("%f, %f, %f \n", C(2,1), C(2,2), C(2,3));
	printf("%f, %f, %f \n", C(3,1), C(3,2), C(3,3));
*/

	//Invert C
	Matrix3x3 c_inverted;

	float denominator = (C(3, 1)*C(1, 2)*C(2, 3) - C(3, 1)*C(1, 3)*C(2, 2) - C(2, 1)*C(1, 2)*C(3, 3) 
		+ C(2, 1)*C(1, 3)*C(3, 2) + C(1, 1)*C(2, 2)*C(3, 3) - C(1, 1)*C(2, 3)*C(3, 2));

	CI(1,1) = (C(2, 2)*C(3, 3) - C(2, 3)*C(3, 2))/denominator; 
	CI(1,2) = (-C(1, 2)*C(3, 3) + C(1, 3)*C(3, 2))/denominator; 
	CI(1,3) = (C(1, 2)*C(2, 3) - C(1, 3)*C(2, 2))/denominator; 

	CI(2,1) = (-C(2, 1)*C(3, 3) + C(3, 1)*C(2, 3))/denominator; 
	CI(2,2) = (-C(3, 1)*C(1, 3) + C(1, 1)*C(3, 3))/denominator; 
	CI(2,3) = (-C(1, 1)*C(2, 3) + C(2, 1)*C(1, 3))/denominator; 

	CI(3,1) = (-C(3, 1)*C(2, 2) + C(2, 1)*C(3, 2))/denominator; 
	CI(3,2) = (-C(1, 1)*C(3, 2) + C(3, 1)*C(1, 2))/denominator; 
	CI(3,3) = (-C(2, 1)*C(1, 2) + C(1, 1)*C(2, 2))/denominator;

/*	printf("\nInverted right Cauchy-Green deformation tensor for tetrahedron %i: \n", me_idx);
	printf("%f, %f, %f \n", CI(1,1), CI(1,2), CI(1,3));
	printf("%f, %f, %f \n", CI(2,1), CI(2,2), CI(2,3));
	printf("%f, %f, %f \n", CI(3,1), CI(3,2), CI(3,3));
*/
	//Find the determinant of the deformation gradient
	float J = X(1, 1)*X(2, 2)*X(3, 3)-X(1, 1)*X(2, 3)*X(3, 2)+X(2, 1)*X(3, 2)*X(1, 3)-
		X(2, 1)*X(1, 2)*X(3, 3)+X(3, 1)*X(1, 2)*X(2, 3)-X(3, 1)*X(2, 2)*X(1, 3);

//	printf("\nDeterminant of the deformation gradient for tetrahedron %i: %f\n", me_idx, J);

	//Calculate stress tensor S from Neo-Hookean Model
	//  S(ij) = mu(delta(ij)-(C(ij)^(-1))^)+lambda^J(J-1)((C^(-1))(ij))

//	float mu = 1007.0f;
//	float lambda = 49329.0f;
	Matrix3x3 s_tensor;

	S(1,1) = mu*(1.0f-CI(1,1)) + lambda*J*(J-1.0f)*CI(1,1);
	S(2,2) = mu*(1.0f-CI(2,2)) + lambda*J*(J-1.0f)*CI(2,2); 
	S(3,3) = mu*(1.0f-CI(3,3)) + lambda*J*(J-1.0f)*CI(3,3);
	S(1,2) = mu*(-CI(1,2)) + lambda*J*(J-1.0f)*CI(1,2);
	S(2,3) = mu*(-CI(2,3)) + lambda*J*(J-1.0f)*CI(2,3);
	S(1,3) = mu*(-CI(1,3)) + lambda*J*(J-1.0f)*CI(1,3); // IS THIS RIGHT?? (3,1) instead?
//	S(1,3) = mu*(-CI(3,1)) + lambda*J*(J-1.0f)*CI(3,1); // IS THIS RIGHT?? (1,3) instead?


/*	printf("\nHyper-elastic stresses for tetrahedron %i: \n", me_idx);
	printf("%f, %f, %f \n", S(1,1), S(1,2), S(1,3));
	printf("%f, %f, %f \n", S(2,1), S(2,2), S(2,3));
	printf("%f, %f, %f \n", S(3,1), S(3,2), S(3,3));
*/
	float4 forces[4];

//	float V = V_0[me_idx];//look up volume
	float V = tex1Dfetch( V0_1d_tex, me_idx );

	//	printf("\nVolume for tetrahedron %i: %f\n", me_idx, V);

	for (int a=1; a<=4; a++) // all 4 nodes
	{
		//Calculate B_L from B_L0 and deformation gradients (a is the node number)

		Matrix6x3 b_tensor;

		B(1,1) = h(a, 1)*X(1, 1);  
		B(1,2) = h(a, 1)*X(2, 1);  
		B(1,3) = h(a, 1)*X(3, 1);  

		B(2,1) = h(a, 2)*X(1, 2);
		B(2,2) = h(a, 2)*X(2, 2);
		B(2,3) = h(a, 2)*X(3, 2);

		B(3,1) = h(a, 3)*X(1, 3);  
		B(3,2) = h(a, 3)*X(2, 3);  
		B(3,3) = h(a, 3)*X(3, 3);  

		B(4,1) = h(a, 2)*X(1, 1) + h(a, 1)*X(1, 2);  
		B(4,2) = h(a, 2)*X(2, 1) + h(a, 1)*X(2, 2);  
		B(4,3) = h(a, 2)*X(3, 1) + h(a, 1)*X(3, 2);  

		B(5,1) = h(a, 3)*X(1, 2) + h(a, 2)*X(1, 3);  
		B(5,2) = h(a, 3)*X(2, 2) + h(a, 2)*X(2, 3);  
		B(5,3) = h(a, 3)*X(3, 2) + h(a, 2)*X(3, 3);

		B(6,1) = h(a, 3)*X(1, 1) + h(a, 1)*X(1, 3);  
		B(6,2) = h(a, 3)*X(2, 1) + h(a, 1)*X(2, 3);  
		B(6,3) = h(a, 3)*X(3, 1) + h(a, 1)*X(3, 3);

/*		printf("\nSubmatrix for a=%i of the stationary strain-displacement matrix for tetrahedron %i: \n", a, me_idx);
		printf("%f, %f, %f \n", B(1,1), B(1,2), B(1,3));
		printf("%f, %f, %f \n", B(2,1), B(2,2), B(2,3));
		printf("%f, %f, %f \n", B(3,1), B(3,2), B(3,3));
		printf("%f, %f, %f \n", B(4,1), B(4,2), B(4,3));
		printf("%f, %f, %f \n", B(5,1), B(5,2), B(5,3));
		printf("%f, %f, %f \n", B(6,1), B(6,2), B(6,3));
*/
		//calculate forces
		float4 force;
		force.x = V*(B(1, 1)*S(1, 1)+B(2, 1)*S(2, 2)+B(3, 1)*S(3, 3)+B(4, 1)*S(1, 2)+B(5, 1)*S(2, 3)+B(6, 1)*S(1, 3));
		force.y = V*(B(1, 2)*S(1, 1)+B(2, 2)*S(2, 2)+B(3, 2)*S(3, 3)+B(4, 2)*S(1, 2)+B(5, 2)*S(2, 3)+B(6, 2)*S(1, 3));
		force.z = V*(B(1, 3)*S(1, 1)+B(2, 3)*S(2, 2)+B(3, 3)*S(3, 3)+B(4, 3)*S(1, 2)+B(5, 3)*S(2, 3)+B(6, 3)*S(1, 3));
		force.w = 0;

		if (length(crop_last_dim(force))<100000 && J>0)
			forces[a-1] = force;
		else
			forces[a-1] = make_float4(0,0,0,0);

	}

/*	printf("\nFor tetrahedron %i: \n", me_idx);
	printf("node1 (%i) force: %f, %f, %f \n", e.x, forces[0].x, forces[0].y, forces[0].z);
	printf("node2 (%i) force: %f, %f, %f \n", e.y, forces[1].x, forces[1].y, forces[1].z);
	printf("node3 (%i) force: %f, %f, %f \n", e.z, forces[2].x, forces[2].y, forces[2].z);
	printf("node4 (%i) force: %f, %f, %f \n", e.w, forces[3].x, forces[3].y, forces[3].z);
*/

	// look up where this tetrahedron is allowed to store its force contribution to a node
	// store force-vector
	pointForces[maxPointForces * e.x + writeIndices[me_idx].x] = forces[0];
	pointForces[maxPointForces * e.y + writeIndices[me_idx].y] = forces[1];
	pointForces[maxPointForces * e.z + writeIndices[me_idx].z] = forces[2];
	pointForces[maxPointForces * e.w + writeIndices[me_idx].w] = forces[3];

//	printf("Max num forces: %i\n", maxPointForces);

//	printf("%i, %i, %i, %i \n", writeIndices[me_idx].x, writeIndices[me_idx].y, writeIndices[me_idx].z, writeIndices[me_idx].w );
}

__global__ void
calculateDrivingForces_k(Point *points, float *masses, float4 *externalForces, unsigned int numPoints)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numPoints)
		return;

	externalForces[me_idx] = make_float4(0, -9820*masses[me_idx], 0, 0); // using millimeters - not meters - thus the factor 1000
//	externalForces[me_idx] = make_float4(0, 0, -9820*masses[me_idx], 0); // using millimeters - not meters - thus the factor 1000

}

__global__ void
applyGroundConstraint_k(Point *points, float4 *displacements, float4 *oldDisplacements,  float lowestYValue, unsigned int numPoints)
{
	int me_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (me_idx>=numPoints)
		return;

	Point me = points[me_idx];
	float4 displacement = displacements[me_idx];

	if ((me.y+displacement.y)<lowestYValue)
	{
		displacements[me_idx].y = lowestYValue - me.y;
		//oldDisplacements[me_idx] = displacements[me_idx];
	}
}

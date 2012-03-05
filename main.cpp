/***************************************************************************/
/*                       CUDA based TLED Solver                            */
/*                     {c} 2008-2010 Karsten Noe                           */
/*                      The Alexandra Institute                            */
/*                   See our blog on cg.alexandra.dk                       */ 
/***************************************************************************/


#include <GL/glew.h>
#include <vector>
#include <GL/glut.h>
#include "TLEDSolver.h"

static int wWidth = 1024, wHeight = 1024;

TetrahedralTLEDState* state;
TetrahedralMesh* mesh;
TriangleSurface* surface;

 
void display()
{
	glPushMatrix();
	for (int i=0; i<40; i++)
	{
		calculateGravityForces(mesh, state); 
		doTimeStep(mesh, state);
		applyFloorConstraint(mesh, state, -30); 
	}
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	display(0,mesh, state, surface);

	glPopMatrix();

    glutPostRedisplay();
	glutSwapBuffers();
 
}

void keyboard( unsigned char key, int x, int y) {
    switch( key) {
        case 27:
        exit (0);
		break;
		case 'n':
			mesh = loadMesh("data/torus_low_res.msh");
			surface = loadSurfaceOBJ("data/torus_low_res.msh.obj");
			precompute(mesh, state, 0.001f, 0.0f, 0.0f, 10007.0f, 5500.0f, 0.5f, 10.0f);
			break;
		/*case 'm':
			mesh = loadMesh("data/torus_low_res.msh");
			surface = loadSurfaceOBJ("torus_low_res.msh.obj");
			precompute(mesh, state, 0.001f, 0.0f, 0.0f, 1007.0f, 49329.0f, 0.5f, 10.0f);
			break;*/
		case 'b':
			mesh = loadMesh("data/torus_low_res.msh");
			surface = loadSurfaceOBJ("data/torus_low_res.msh.obj");
			precompute(mesh, state, 0.001f, 0.0f, 0.0f, 100007.0f, 493.0f, 0.5f, 10.0f);
			break;
		case 'v':
			mesh = loadMesh("data/torus_low_res.msh");
			surface = loadSurfaceOBJ("data/torus_low_res.msh.obj");
			precompute(mesh, state, 0.001f, 0.0f, 0.0f, 100007.0f, 200.0f, 0.5f, 10.0f);
			break;
        default: break;
    }
}

void reshape(int x, int y) {
    wWidth = x; wHeight = y;
   glViewport(0, 0, x, y);
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   glFrustum(-4, 4, -4, 4, 5, 200);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glEnable(GL_LIGHTING);
   //float redMaterial[] = { 1, 1, 1, 1 };
   float lightmat[] = { 0.7f, 0.7f, 0.7f, 0.001f };
   float lightamb[] = { 0.7f, 0.7f, 0.7f, 0.001f };
   float lightpos[] = { 0.f, 0.f, 1500.f, 1.f};
   glEnable(GL_LIGHT0);
   glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
   glLightfv(GL_LIGHT0, GL_DIFFUSE, lightmat);
   gluLookAt(-40,0,-40,0,0,0,0,1,0);
   glutPostRedisplay();
}


int main(int argc, char** argv) {

    glutInit(&argc, argv);

	printf("***********************************************************************\n");
	printf("TLED solver CUDA implementation by Karsten Noe.\n");
	printf("Press v,b,n or m to reload model and use different material parameters.\n");
	printf("***********************************************************************\n");

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(wWidth, wHeight);
    glutCreateWindow("TLED");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutReshapeFunc(reshape);

    glewInit();

    if (! glewIsSupported(
        "GL_ARB_vertex_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return -1;
    }

	state = new TetrahedralTLEDState(); 

	mesh = loadMesh("data/torus_low_res.msh");
	surface = loadSurfaceOBJ("data/torus_low_res.msh.obj");
	precompute(mesh, state, 0.001f, 0.0f, 0.0f, 1007.0f, 49329.0f, 0.5f, 10.0f);

	atexit(cleanupDisplay);
    glutMainLoop();

    return 0;

}

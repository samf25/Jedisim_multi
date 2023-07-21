#include <stdio.h>
#include <string.h>
#include <math.h>
#include "fitsio.h"
#include <stdbool.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>
#include <unistd.h>

/*to compile, run:
 *gcc -I/users/sferrar2/data/sferrar2/CWork/include -O3 jeditransform.c /users/sferrar2/data/sferrar2/CWork/lib/libcfitsio.a  /oscar/runtime/opt/zlib/1.2.11/lib/libz.a -o jeditransform_s -lm -lpthread
 *
 * Where the -I arg is where the cfitsio include folder path
 * The second path is to the static cfitsio library
 * The third path is to a static library that provides required functions for cfitsio (it's availible to all of oscar).
 * When setting up cfitsio, use the --enable-reentrant command with configure so it will work with multithreading.
 */

//Transform Variables
#define PI  3.14159         //because C doesn't have it built-in for some silly reason
#define EPSILON 0.0000001   //a very small number
#define MAG0 30.0           //magnitude zero point
//#define MAG0 26.66

//Distort Variables
#define DR      10          //lens table entries per pixel
#define C       300000      //speed of light in km/s
#define NSUBPX  4           //number of subpixels in each dimension
#define BNSUBPX 2           //log base two of the number of subpixels for bitshift magic
#define OMEGA_M 0.315       //matter density of the universe today
#define OMEGA_D 0.685       //dark energy density of the universe
#define OPZEQ   3391        //1 + Z_eq
#define C_H_0   4424.778    //c [km/s] / Hubble constant today [km/(s*Mpc)] with H_0 = 67.80
#define G       4.302       //Gravitational constant [10^{-3} (pc/Solar mass) (km/s)^2]
#define H0      67.80       //Planck value for Hubble constant for present day [km/(Mps*s)]
#define EP      0           //epsilon
 
//Paste Variables
#define NUMBANDS 2


char *help[] = {
  "Takes in a catalog of images, and produces a FITS image for each entry, transformed to the correct specifications.",
  "Usage: jeditransform catalog distort_list",
  "Arguments:	catalog - text file containing galaxy catalog",
  "		lenses - file containing les parameters",
  "		nthreads - the maximum number of active threads",
  "		x - width of the image MUST BE AN INTEGER MULTIPLE OF 4096",
  "		y - height of the image MUST BE AN INTEGER MULTIPLE OF 4096",
  "		scale - pixel scale in arcseconds per pixel",
  "		zl - lens redshift",
  "		outfile - name of output file",
  "		config - name of config file",
  "catalog file: image x y angle redshift old_mag old_r50 new_mag new_r50 [tab separated]",
  "		image - file path for the base galaxy postage stamp image",
  "		x - x coordinate for the image center",
  "		y - y coordinate for the image center",
  "		angle - angle through which to rotate the input postage stamp",
  "		redshift - redshift for the galaxy",
  "		pixscale - pixel scale for the galaxy (arcseconds per pixel)",
  "		old_mag - magnitude of the base galaxy postage stamp image",
  "		old_r50 - r50-type radius of the base galaxy postage stamp image",
  "		new_mag - magnitude that the galaxy should have",
  "		new_r50 - r50-type radius that the galaxy should have",
  "Lens parameter file: x y rho0 Rs zl",
  "		x - x center of lens (in pixels)",
  "		y - y center of lens (in pixels)",
  "		type - type of mass profile",
  "			1. Singular isothermal sphere",
  "			2. Navarro-Frenk-White profile",
  "3. NFW constant distortion profile for grid simulations",
  "		p1 - first profile parameter",
  "			1. sigma_v [km/s]",
  "			2. M200 parameter [10^14 solar masses]",
  "3. Distance to center in px. M200 fixed at 10 default, which can be modified in case 3",
  "		p2 - second profile parameter",
  "			1. not applicable, can take any numerical",
  "			2. c parameter [unitless]",
  "			3. c parameter [unitless]",

  0};

//all the information needed to describe a single galaxy
typedef struct{
  char		*image;		//file path for the base galaxy postage stamp image",
  float		x;		//x coordinate for the image center",
  long int	xembed;		//x coord of lower left pixel where galaxy should be embedded",
  float		y;		//y coordinate for the image center",
  long int	yembed;		//y coord of lower left pixel where galaxy should be embedded",
  long int	nx;		//width of galaxy in pixels",
  long int	ny;		//height of galaxy in pixels",
  int		naxis;		//fits file dimension",
  float		angle;		//angle through which to rotate the input postage stamp",
  float		redshift;	//redshift for the galaxy",
  float		pixscale;	//pixelscale for the galaxy",
  float		old_mag;	//magnitude of the base galaxy postage stamp image",
  float		old_r50;	//r50-type radius of the base galaxy postage stamp image",
  float		new_mag;	//magnitude that the galaxy should have",
  float		new_r50;	//r50-type radius that the galaxy should have",
} galaxy;

typedef struct {
  float       x;          // x coord of lens (pixels)
  float       y;          // y coord of lens (pixels)
  int         type;       // Identifies mass profile 
  float       p1;         // mass profile parameter 1
  float       p2;         // mass profile parameter 2
  long int    nr;         // Number of entries in the distortion table (= max radius*DR)
  float       *table;     // Distortion table: M_enclosed(r) where r is measured in pixels
} lens;

typedef struct {
  float       xmin;        //smallest x value
  float       xmax;        //largest x valu  float       ymin;        //smallest y value
  float       ymin;        //smallest y value
  float       ymax;        //largest y value
} rect;

//internal functions
float bilinear_interp(float x, float y, int xmax, int ymax, float *img);
void get_alpha(long int x, long int y, int nlenses, lens* lenses, float* alphax, float* alphay);
float angular_di_dist(float z1, float z2);
void print_rect(rect *r);

//setup
void setupTransform(int argc, char *argv[]);
void setupDistort(int argc, char *argv[]); 
void setupPaste();

//threads
float *threadTransform(int g);
float *threadDistort(int gal, float *galimage);
void *threadPaste(int gal, float *galimage);

//I made these global so that I wouldn't have to pass it into galThread
galaxy  *galaxies;		//array of galaxies to read gallist into
sem_t	mysem;			//semaphore to limit thread count
char	**paths;		//paths of the files
int	count;			//number of files
pthread_mutex_t *locks;		//mutex locks to prevent thread overlap
struct	timespec ts = {0, 500};	//a very short time to sleep
float	zl;			//lens redshift
lens	*lenses;		//array of lens structs

//Declare Variables
#define ngr 4         //number of grids
int         g = 8;          //exponential factor to grow by between grid sizes
int         b[4] = {3, 6, 9, 12};       //array of log_2 of grid square sizes for bitshit ops
int         ngx[4], ngy[4];         //width and height of grids.
long int    nx, ny;         //width and height of the image
long int    ngalaxies;      //number of galaxies
long int    nlenses;        //number of lenses
float       scale;          //pixel scale in arcseconds per pixel
rect        *grids[ngr];    //array of pointers to the grids

//Paste variables
float		*fimage;		//final image after pasting
pthread_mutex_t	*bandlocks;		//mutex locks for the final image

//actions to be preformed on each galaxy
void *runThread(void *argv) {
	float		*output1, *output2;
	int		*num = (int *)argv;
	int		g = *num;
	printf("%s\n",galaxies[g].image);
	//run transform
	output1 = threadTransform(g);

	//run distort
	output2 = threadDistort(g, output1);

	//run paste
	threadPaste(g, output2);
	
	//update user
	if (g % 10000 == 0) {fprintf(stdout, "Transformed, Distorted, and Pasted image %i.\n", g);}

	//indecate completion
	sem_post(&mysem);
}


int main(int argc, char *argv[]){
  int		nthreads;	//maximum number of active threads
  int     	status = 0;	//CFITSIO status
  long int	naxis = 2;	//CFITSIO variables
  fitsfile 	*ffptr;		//pointer to final image as fits
 
  /* Print help */
  if ((argc != 10) || (argv[1][0] == '^')) {
    int i;
    for (i = 0; help[i] != 0; i++)
      fprintf (stderr, "%s\n", help[i]);
    exit (1);
  }

  sscanf(argv[3], "%d %[^\t\n]", &nthreads);
  fprintf(stdout, "Max Threads: %d\n", nthreads);

  //set up transform
  setupTransform(argc, argv);

  //set up distort
  setupDistort(argc, argv);

  //set up paste
  setupPaste();

  pthread_t tid[ngalaxies]; //thread id
  pthread_attr_t attr; //attributes
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED); //Set all threads to be detached so we don't have to take the time to join them at the end

  sem_init(&mysem, 0, nthreads); //set up semaphore
  int glst[ngalaxies]; //to pass index into each thread (needed because g changes and we are passing pointer)  
  int g;
  //loop over each image, rotating, scaling, photoscaling, and "cutting out" each one
  for(g = 0; g < ngalaxies-1; g++){
    glst[g] = g;
    sem_wait(&mysem); //prevents too many threads
    pthread_create(&tid[g], &attr, runThread, (void *)&glst[g]); //create a thread
  }
  //Run the last file in this thread and sleep to allow all to finish before exiting.
  glst[ngalaxies-1]=ngalaxies-1;
  runThread((void *)&glst[ngalaxies-1]);
  sleep(12);
 
  //write to output
  fprintf(stdout, "Writing final image %s. This may take several minutes", argv[8]);
  fits_create_file(&ffptr, argv[8], &status);
  long int fnaxes[2] = {nx, ny};	//Dimensions of the final image
  long int opixel[2] = {1, 1};	//First pixel to read in
  fits_create_img(ffptr, FLOAT_IMG, naxis, fnaxes, &status);
  fits_write_pix(ffptr, TFLOAT, opixel, nx*ny, fimage, &status);
  fits_close_file(ffptr, &status);
  fits_report_error(stderr, status);

  //free memory
  sem_destroy(&mysem);
  pthread_attr_destroy(&attr);
  free(paths);
  free(locks);
  free(bandlocks);
  free(galaxies);
  free(fimage);
  
  fprintf(stdout, "Jeditransform, distort, and paste complete.");
  return 0;
}


//given a floating point location in an image, returns its pixel value by doing a bilinear interpolation
float bilinear_interp(float x, float y, int xmax, int ymax, float *img){
  int xi = (int) x, yi = (int) y;
  if(xi >= 0 && xi < xmax-1 && yi >= 0 && yi < ymax-1){
    float a = img[xi + ymax*yi];            //f(xi,yi)
    float b = img[xi+1+ymax*yi];      //f(xi+1,yi)
    float c = img[xi + ymax*(yi+1)];            //f(xi, yi+1)
    float d = img[xi+1+ymax*(yi+1)];      //f(xi+1, yi+1)
    float xf = x -xi, yf = y-yi;
    return a*(1-xf)*(1-yf) + b*xf*(1-yf) + c*(1-xf)*yf + d*xf*yf;
  } else
    return 0;
}

//given a  pixel (x,y), and the list of lenses,
//returns the vector (alpha_x,alpha_y) at that pixel
void get_alpha(long int x, long int y, int nlenses, lens* lenses, float* alphax, float* alphay){
  long int         nlens;      //counter
  for(nlens = 0; nlens < nlenses; nlens++){
    float dx = ((float) x)/NSUBPX - lenses[nlens].x;
    float dy = ((float) y)/NSUBPX - lenses[nlens].y;

    //the table entry in the lens table is the dist from lens center to Point of Interest
    //times the table interval (DR per integer), rounded to an integer
    // (int) floating_point + 0.5 is a quick way to round a float to an int
    long int rad = (long int) (DR*sqrt(dx*dx +dy*dy)+0.5);
    //we want to add alpha(r) * (x/r) but the table stores alpha(r)/r to speed things up
    //where r is measured in px/NSUBPX
    *alphax += lenses[nlens].table[rad]*dx;
    *alphay += lenses[nlens].table[rad]*dy;
  }
}


//given two redshifts, 
//returns the angular diameter distance between them in Mpc for a set cosmology
float angular_di_dist(float z1, float z2){
  float dist = 0, z;
  float dz = 0.001;
  for(z = 1+z1; z < 1+z2; z+= dz){
    dist += dz/sqrt(OMEGA_M*(1.0+(z/OPZEQ))*(z*z*z) + OMEGA_D);
  }
  dist = C_H_0*dist/z;
  return dist;
}

//for debugging
void print_rect(rect* r){
  fprintf(stdout,"xmin: %f\t ymin: %f\t xmax: %f\t  ymax: %f\n", (*r).xmin, (*r).ymin, (*r).xmax, (*r).ymax);
}

//Transform the galaxy
float *threadTransform(int g) {
    fitsfile    *galfptr;             //fits pointers for the input file
    int         status = 0, naxis = 0;          //cfitsio status counter & image dimensionality
    long        galnaxes[2], tgalnaxes[2];      //cfitsio image dimensions array
    long        fpixel[2] = {1,1}, lpixel[2];   //cfitsio first/last pixel to read in/out
    float       *galaxy, *tgal, *pgal;          //arrays for galaxy input image, transformed image, postage transformed image

    //find file index looping through all posible until found
    int i=0;
    int index=-1;
    while (i < count) {
      if (strcmp(paths[i],galaxies[g].image) == 0) {
        index = i;
        break;
      }
      i++;
    }
    //lock the file being opened
    pthread_mutex_lock(&locks[index]);

    //open input image
    fits_open_file(&galfptr, galaxies[g].image, READONLY, &status);
    fits_get_img_dim(galfptr, &naxis, &status);
    fits_get_img_size(galfptr, 5, galnaxes, &status);

    //get image dimensions
    if(status){
      fits_report_error(stderr,status);
      exit(1);
    }

    //allocate enough memory for the galaxy and transformed galaxy images
    //fprintf(stdout, "Allocating memory for galaxy %i image.\n", g);
    galaxy = (float *) calloc (galnaxes[0] * galnaxes[1], sizeof(float));
    tgal = (float *) calloc (galnaxes[0] * galnaxes[1], sizeof(float));
    if (galaxy == NULL){
      fprintf(stderr, "Error: cannot allocate memory for galaxy %i image.\n", g);
      exit(1);
    }

    //read in the image
    //fprintf(stdout, "Reading in image.\n");
    if(fits_read_pix(galfptr, TFLOAT, fpixel, galnaxes[0]*galnaxes[1], NULL, galaxy, NULL, &status)){
      fprintf(stderr, "Error: cannot read in galaxy %i image.\n", g);
      fits_report_error(stderr, status);
      exit(1);
    }
    fits_close_file(galfptr, &status);

    //unlock file
    pthread_mutex_unlock(&locks[index]);

    //find magnifications
    float   scale = galaxies[g].old_r50/galaxies[g].new_r50;

    float c = cosf(galaxies[g].angle*PI/180);   //pre-compute cos
    float s = sinf(galaxies[g].angle*PI/180);   //and sin

    //variables for image transformation
    long int    xmin = galnaxes[0], xmax = -1, ymin = galnaxes[1], ymax = -1;   //information bounding box in transormed image
    long int    xc = 1+(galnaxes[0] >> 1), yc = 1+(galnaxes[0] >> 1);           //x and y center pixels
    float       newx, newy, oldx, oldy;                                         //x and y positions relative to the image center, pre- and post- transform
    long int    x, y;                                                           //counters

    float total = 0;        //the total value of all the pixels in the image

    //transform each pixel

    for(x = 0; x < galnaxes[0]; x++){
      for(y = 0; y < galnaxes[1]; y++){
	//find pixel relative to center
	oldx = (float) (x - xc);
	oldy = (float) (y - yc);

	//rotation and dilation matrix
	newx = scale * (oldx*c - oldy*s) + xc;
	newy = scale * (oldx*s + oldy*c) + yc;

	//do interpolation and find new magnitude
	float value = bilinear_interp(newx, newy, galnaxes[0], galnaxes[1], galaxy);
	if(value > EPSILON){
	  if(x < xmin)
	    xmin = x;
	  if(x > xmax)
	    xmax = x;
	  if(y < ymin)
	    ymin = y;
	  if(y > ymax)
	    ymax = y;
	}
	tgal[y*galnaxes[1]+x] = value;
	total += value;
      }
    } 

    float mag = MAG0 - 2.5*log10f(total);
    float photoscale = pow(10.0,(mag - galaxies[g].new_mag)/2.5);

    //find the size of the smallest possible postage stamp needed
    tgalnaxes[0] = (xmax+1)-xmin;
    tgalnaxes[1] = (ymax+1)-ymin;

    //make smallest possible postage stamp galaxy, and adjust brightness to desired magnitude
    pgal = (float *) calloc(tgalnaxes[0]*tgalnaxes[1], sizeof(float));
    for(x = xmin; x < xmax + 1; x++)
      for(y = ymin; y < ymax + 1; y++)
	pgal[(y-ymin)*tgalnaxes[0] + (x-xmin)] = photoscale*tgal[y*galnaxes[1] + x];

    //specify the coordinates where the lower left-most pixel of this image should be embedded in the galaxy field
    long int    xembed = (long int) (0.5 + galaxies[g].x - (1+(float) tgalnaxes[0])/2);
    long int    yembed = (long int) (0.5 + galaxies[g].y - (1+(float) tgalnaxes[1])/2);

    //add info for distort
    galaxies[g].xembed = xembed;
    galaxies[g].yembed = yembed;
    galaxies[g].nx = tgalnaxes[0];
    galaxies[g].ny = tgalnaxes[1];
    galaxies[g].naxis = naxis;
  
    free(galaxy);
    free(tgal);

    return pgal;
}

//distort the galaxy
float *threadDistort(int gal, float *galimage) {
    //fits variables
    fitsfile    *outfptr;         //FITS file pointers for the output images
    int         status = 0;      //error reporting
    long        naxes[2] = {galaxies[gal].nx, galaxies[gal].ny}, fpixel[2] = {1,1};//variables for FITS to read in files
    float       *outimage;       //output image as array
    bool        *grids_t[ngr];  //array of pointers to a true table for each grid, true means that gridsquare contains information

    float     prefactor = angular_di_dist(zl,galaxies[gal].redshift)/angular_di_dist(0,galaxies[gal].redshift);          //the quantity D_ls/D_s

    rect gal_box = {(float) galaxies[gal].xembed, (float) (galaxies[gal].xembed + galaxies[gal].nx), (float) galaxies[gal].yembed, (float) (galaxies[gal].yembed + galaxies[gal].ny)};

    //allocate memory for the grid_t truth tables
    int gr;
    for(gr = 0; gr< ngr; gr++)
      grids_t[gr] = (bool *) calloc(ngx[gr] * ngy[gr], sizeof(bool));

    //we want to compute the smallest box that contains all the information for the finest grid
    int         ixmax = -1;          //convention: lowest row after all rows containing information
    int         ixmin = ngx[0];     //convetion: lowest row that contains information
    int         iymax = -1;          //convention: lowest column after all rows containing information
    int         iymin = ngy[0];     //convetion: lowest column that contains information

    for(gr = ngr; gr > 0; gr--){
      //loop over all grid squares
      long int     row, col, srow, scol;  //variables to loop over grids and sub-grids
      long int     gs_wd = (ngr == gr ? nx : g << (b[0]*gr));
      long int     gs_ht = (ngr == gr ? ny : g << (b[0]*gr));
      long int     s_gs_s= g << (b[0]*(gr-1));        //sub-gridsquare sieve length
      int subrows_per_row = (ngr == gr ? (nx >> b[gr-1]) : g);
      int subcols_per_col = (ngr == gr ? (nx >> b[gr-1]) : g);

      for(row = 0; row < (ngr == gr ? 1 : ngx[gr]); row++){
	for(col = 0; col < (ngr == gr ? 1 : ngy[gr]); col++){
	  //check if we want ot look at this sub-gridsquare
	  //always yes for the top level, otherwise look it up in the table
	  if(ngr == gr || grids_t[gr][col*ngy[gr] + row]){
	    //loop over all the sub-gridsquares
	    //fprintf(stdout,"examining grid square (%i, %i)\n", row, col);
	    for(srow = 0; srow < subrows_per_row; srow++){
	      for(scol = 0; scol < subcols_per_col; scol++){
		//fprintf(stdout,"(%i, %i)\n", gs_wd*row+s_gs_s*srow, gs_ht*col+s_gs_s*scol);
		/*Now we want to see if the gal_box rectangle and the rectangle in the source plane from which it is possible for this grid square to draw information have an intersection. This is done using the efficient method from http://stackoverflow.com/questions/306316/determine-if-two-rectangles-overlap-each-other. An epsilon margin is added around gal_box, in case of rounding issues or other losses of precision.

		    The rect in the source plane is given by (note reversal of alpha min and max)
		      x'_min = x_min - prefactor * alpha.xmax
		      x'_max = x_max - prefactor * alpha.xmin */

		long int index = ((col << b[0]) + scol)*ngy[gr-1] + (row << b[0]) + srow;

		rect *alpha_box = &grids[gr-1][index];
		//print_rect(alpha_box);
		/* 
		      fprintf(stdout,"CHECK: gr: %i\t row: %i\t col: %i\t x1: %i\t y1: %i\t x2: %i\t y2: %i\n", gr-1, (row << b[0]) + srow, (col << b[0]) + scol,((row << b[0]) + srow) << b[gr-1], ((col << b[0]) + scol) <<b[gr-1],((row << b[0]) + srow+1) << b[gr-1], ((col << b[0]) + scol+1) << b[gr-1]  );
		         print_rect(alpha_box);
			    fprintf(stdout,"xmin: %f\tymin: %f\txmax: %f\tymax: %f\n",gs_wd*row+s_gs_s*srow - prefactor*(*alpha_box).xmax, gs_ht*col+s_gs_s*scol - prefactor*(*alpha_box).ymax, gs_wd*row+s_gs_s*(srow+1) - prefactor*(*alpha_box).xmin,gs_ht*col+s_gs_s*(scol+1) - prefactor*(*alpha_box).ymin);
			       print_rect(&gal_box);
		*/
		//using 4 if's for readability, but one would suffice
		//let rect A be the source box and B be the gal box
		//check if A's Left Edge is to the left of B's right edge AND
		if(gs_wd*row+s_gs_s*srow - prefactor*(*alpha_box).xmax < gal_box.xmax){
		  //check if A's right edge is to right of B's left edge AND

		  if(gs_wd*row+s_gs_s*(srow+1) - prefactor*(*alpha_box).xmin > gal_box.xmin){
		    //check if A's top is above B's bottom AND
		    if(gs_ht*col+s_gs_s*(scol+1) - prefactor*(*alpha_box).ymin > gal_box.ymin){
		      //check if A's bottom is below B's Top
		      if(gs_ht*col+s_gs_s*scol - prefactor*(*alpha_box).ymax < gal_box.ymax){
			grids_t[gr-1][index] = TRUE;
			/*fprintf(stdout,"TRUE:gr: %i\t row: %i\t col: %i\t x1: %i\t y1: %i\t x2: %i\t y2: %i\n", gr-1, (row << b[0]) + srow, (col << b[0]) + scol,((row << b[0]) + srow) << b[gr-1], ((col << b[0]) + scol) <<b[gr-1],((row << b[0]) + srow+1) << b[gr-1], ((col << b[0]) + scol+1) << b[gr-1]  );
			    print_rect(alpha_box);
			    fprintf(stdout,"xmin: %f\tymin: %f\txmax: %f\tymax: %f\n",gs_wd*row+s_gs_s*srow - prefactor*(*alpha_box).xmax, gs_ht*col+s_gs_s*scol - prefactor*(*alpha_box).ymax, gs_wd*row+s_gs_s*(srow+1) - prefactor*(*alpha_box).xmin,gs_ht*col+s_gs_s*(scol+1) - prefactor*(*alpha_box).ymin);*/
			//find information bounding box for finest grid
			if(gr == 1){
			  if((row << b[0]) + srow > ixmax)
			    ixmax = (row << b[0]) + srow;
			  if((row << b[0]) + srow < ixmin)
			    ixmin = (row << b[0]) + srow;

			  if((col << b[0]) + scol > iymax)
			    iymax = (col << b[0]) + scol;
			  if((col << b[0]) + scol < iymin)
			    iymin = (col << b[0]) + scol;
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }

    ixmax++;    //we want one beyond the max
    iymax++;    //same
    //so many braces!
    /*j=0;
        gr =0;
	  long int row1, col1;
	    for(row1 = 0; row1 < ngx[gr]; row1++)
	      for(col1 = 0; col1 < ngy[gr]; col1++)
	        if(grids_t[gr][col1*ngy[gr]+row1]){
		  j++;
		    fprintf(stdout,"%i: (%i, %i)\n", j, row1, col1);
		    }*/
    //fprintf(stdout,"Information box: xmin: %i; xmax: %i; ymin: %i; ymax: %i.\n",ixmin, ixmax, iymin, iymax);


    //if there's no grid squares that contain any information, set up the min and max so that the rest of the program does nothing
    if(ixmin > ixmax || iymin > iymax){
      //fprintf(stdout,"Warning: galaxy %i is completely of the screen.\n", gal);
      ixmin = 0;
      ixmax = 1;
      iymin = 0;
      iymax = 1;
    }


    /*Now we want to loop over each of the grid squares of the finest grid and see which ones contain information. We then determine the smallest box which contains all of them, and make a postage stamp output image which contains all information for the larger image. For each grid square that contains information, we want to loop over each individual pixel and break it into subpixels. For each subpixel, compute alpha and, via the lens equation beta = theta - alpha(theta), return the value of the pixel in the source plane which the lens equation points to. The value of the observed image at that pixel will be the average of all the subpixels.*/ 

    long int        onaxes[2] = {(ixmax - ixmin) << b[0],(iymax - iymin) << b[0]};      //(width,height) of output image in pixels.
    long int        ofx  = ixmin << b[0];                //x pixel of lower left corner of output image
    long int        ofy  = iymin << b[0];                //y pixel of lower left corner of output image


    //allocate memory for the output image
    outimage = (float *) calloc(onaxes[0]*onaxes[1], sizeof(float));
    if(outimage == NULL){
      fprintf(stderr,"Error allocating memory for output image %i.\n", gal);
      exit(1);
    }
    //fprintf(stdout,"onaxes: (%i, %i)\n", onaxes[0], onaxes[1]); 

    //set up some counters
    long int    row, col;       //row and column in the smallest grid
    long int    ox, oy;         //output image pixels
    long int    x, y;           //calculated source x and y pixels
    int         sx, sy;            //subpixels


    //Pretabulate some sub-pixel information to avoid some floating point calculations. dy would look the same as dx, so we don't need to bother.
    float dx[NSUBPX];
    for(sx = 0; sx < NSUBPX; sx++)
      dx[sx] = sx*(1.0/NSUBPX) + (1.0/(2*NSUBPX)) - 0.5;



    //loop over the finest grid 
    for(row = ixmin; row < ixmax; row++){
      for(col = iymin; col < iymax; col++){
	if(grids_t[0][col*ngy[0]+row]){
	  //calculate every pixel
	  for(ox = (row << b[0]); ox < ((row+1) << b[0]); ox++){
	    for(oy = (col << b[0]); oy < ((col+1) << b[0]); oy++){
	      float     outvalue = 0;   //output value of this pixel
	      //find the value for each subpixel

	      for(sx = 0; sx < NSUBPX; sx++){
		for(sy = 0; sy < NSUBPX; sy++){
		  //get alpha vector
		  float alpha_x = 0, alpha_y = 0;
		  get_alpha((ox << BNSUBPX) + sy, (oy << BNSUBPX) +sy, nlenses, lenses, &alpha_x, &alpha_y);
		  //calculate
		  x = (long int) (ox + (((float) sx)/NSUBPX) - prefactor*alpha_x); 
		  y = (long int) (oy + (((float) sy)/NSUBPX) - prefactor*alpha_y);

		  if(x >= gal_box.xmin && x < gal_box.xmax && y >= gal_box.ymin && y < gal_box.ymax){
		    // the galimage array doesn't start at the origin, so we have to shift over
		    x = x - (long int) gal_box.xmin; 
		    y = y - (long int) gal_box.ymin;
		    // if(y > 295 && y < 305 && x > 295 && x < 305) fprintf(stdout,"(%i, %i): prefactor: %f\t x shift: %f\t y shift: %f\n", ox, oy, prefactor, prefactor*alpha_x, prefactor*alpha_y);
		    //fprintf(stdout, "(%i, %i)\n", x, y);
		    if(y*naxes[0]+x > naxes[0]*naxes[1]){
		      fprintf(stderr,"Error: index too large. index: %li; max index: %li; ox: %li; oy: %li; x: %li; y: %li; naxes[0]: %li; naxes[1]: %li\n", y*naxes[0]+x, naxes[0]*naxes[1], ox, oy, x, y, naxes[0], naxes[1]); 
		      exit(1);
		    } else
		      outvalue += galimage[y*naxes[0] + x];
		  }
		}
	      }
	      //outvalue = galimage[(ox-ofx)*naxes[0]+(oy-ofy)]; //replicate input image for debugging

	      outimage[(oy -ofy)*onaxes[0] + (ox-ofx)] =  outvalue/(NSUBPX*NSUBPX); 
	    }
	  }
	}
      }
    }

    //redefine galaxy size
    galaxies[gal].nx = onaxes[0];
    galaxies[gal].ny = onaxes[1];

    //release memory
    free(galimage);
    for(gr = 0; gr< ngr; gr++)
      free(grids_t[gr]);
    return outimage;
}

void *threadPaste(int g, float *galimage) {
  int locknum1 = galaxies[g].yembed * NUMBANDS /nx; //The band the bottom of the image is in
  int locknum2 = (galaxies[g].yembed - galaxies[g].ny) * NUMBANDS /nx; //The band the top of the image is in
  //lock the band(s) being written
  pthread_mutex_lock(&bandlocks[locknum1]);
  if (locknum1 != locknum2) {
    pthread_mutex_lock(&bandlocks[locknum2]);
  }
  //Loop over image and add it to the final image in the correct location
  int col;
  int row;
  for (col = 0; col < galaxies[g].nx; col++) {
    for (row = 0; row < galaxies[g].ny; row++) {
      fimage[(col + galaxies[g].xembed) + nx * (row + galaxies[g].yembed - galaxies[g].ny)] += galimage[col + galaxies[g].nx * row];
    }
  }
  //unlock the band(s) being written
  pthread_mutex_unlock(&bandlocks[locknum1]);
  if (locknum1 != locknum2) {
    pthread_mutex_unlock(&bandlocks[locknum2]);
  }
  free(galimage); //free memory
}

void setupTransform(int argc, char *argv[]) {
  //declare variables
  char	gallist_path[1024]; //file path for the list of galaxies
  FILE	*gallist_file, *config_fp;      //galaxy list and config file
  char	buffer1[1024], buffer[1024], buffer3[1024]; //buffers for reading in strings
  char	*buffer2; //buffer for reading in strings

  //parse command line input
  sscanf(argv[1], "%[^\t\n]", &gallist_path);
  fprintf(stdout, "gallist_path: %s\n", gallist_path);

  //parse gallist file
  if((gallist_file = fopen(gallist_path, "r")) == NULL){
    fprintf(stderr,"Error: could not open galaxy list file \"%s\"\n", gallist_path);
    exit(1);
  }
  fprintf(stdout,"Opened galaxy list file \"%s\".\n", gallist_path);

  //first count the number of galaxies
  ngalaxies = 0;
  galaxy temp;
  while(fscanf(gallist_file, "%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f", &buffer1, &temp.x, &temp.y, &temp.angle, &temp.redshift, &temp.pixscale, &temp.old_mag, &temp.old_r50, &temp.new_mag, &temp.new_r50) > 0)
    ngalaxies++;

  if(ngalaxies ==0){
    fprintf(stderr,"Error: found 0 galaxies in galaxy list \"%s\"\n", gallist_path);
    exit(1);
  }
  fprintf(stdout,"Found %i galaxies in galaxy list \"%s\"\n", ngalaxies, gallist_path);

  //allocate memory for the galaxies
  galaxies = (galaxy *) calloc(ngalaxies, sizeof(galaxy));
  if(galaxies == NULL){
    fprintf(stderr, "Error: could not allocate memory for galaxy list.");
    exit(1);
  }

  //read in the galaxies
  rewind(gallist_file);
  int     g;  //galaxies counter
  for(g = 0; g < ngalaxies; g++) {
    fscanf(gallist_file, "%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f", &buffer1, &galaxies[g].x, &galaxies[g].y, &galaxies[g].angle, &galaxies[g].redshift, &galaxies[g].pixscale, &galaxies[g].old_mag, &galaxies[g].old_r50, &galaxies[g].new_mag, &galaxies[g].new_r50);
    galaxies[g].image = (char *) calloc(strlen(buffer1)+1, sizeof(char));
    if(galaxies[g].image == NULL){
      fprintf(stderr,"Error: could not allocate memory for galaxy  %i.\n", g);
      exit(1);
    }
    strcpy(galaxies[g].image, buffer1);
  }
  fclose(gallist_file);
  for (g=0;g<ngalaxies;g++) {
    printf("%s\n",galaxies[g].image);
  }
  fprintf(stdout,"Read in %i galaxies.\n", ngalaxies);

  //parse config file for stamp addresses 
  config_fp = fopen(argv[9],"r");
  if(config_fp == NULL){
    fprintf(stderr,"Error: cannot open config file.");
    exit(1);
  }
 
  int nimage = 0;
  while(fgets(buffer, 1024, config_fp) != NULL){
    //read in whatever isn't a comment
    if(buffer[0] != '#'){
      buffer2 = strtok(buffer,"#\n");
      sscanf(buffer2,"%[a-z_]=",buffer3);
      //image settings
      if(strcmp(buffer3,"num_source_images")==0){
        sscanf(buffer2,"num_source_images=%li",&count);
        paths = (char **) calloc(count, sizeof(char *));
        if(paths == NULL){
          fprintf(stderr,"Error: could not allocate list of source images.\n");
          exit(1);
        }
      } else if(strcmp(buffer3,"image")==0 && nimage < count){
          if(paths == NULL){
            fprintf(stderr, "Error: the parameter 'num_galaxies' must come before any 'image' parametersn");
            exit(1);
          }
          char temp[1024];
          sscanf(buffer2, "image=\"%[^\"]", temp);
          paths[nimage] = (char*) calloc(strlen(temp)+1, sizeof(char));
          if(paths[nimage] == NULL){
            fprintf(stderr,"Error: could not allocate source image struct %i.\n", nimage);
            exit(1);
          }
          strcpy(paths[nimage],temp);
          nimage++;
        }
    }
  }
  fclose(config_fp); 
 
  //make the mutex array (These will prevent the threads from trying to pull from the same file at once)
  locks = calloc(count, sizeof(pthread_mutex_t));
  if (locks == NULL) {
    fprintf(stderr, "Error: could not allocate memory for mutex locks");
    exit(1);
  }

  //initialize mutex array used to look input files
  for(g = 0; g < count; g++) {
    if (pthread_mutex_init(&locks[g], NULL) != 0) {
        fprintf(stderr, "Error: mutex init has failed\n");
        exit(1);
    }
  }
  fprintf(stdout, "Number of Galaxies used: %d.\n", count);
}

void setupDistort(int argc, char *argv[]) {
  FILE        *galaxyfile;    //galaxy list file
  FILE        *lensfile;      //lens list file
  galaxy      tempgal;        //temporary galaxy for reading in galaxies
  lens        templens;       //temporary lens for reading in lenses
  int 	      nthreads;	      //maximum number of threads running at once

  /* parse command line input*/
  sscanf(argv[4], "%li", &nx);
  sscanf(argv[5], "%li", &ny);
  sscanf(argv[6], "%f", &scale);
  sscanf(argv[7], "%f", &zl);


  //check that the width and height of the image are multiples of 4096
  if((nx & (4096-1)) != 0 || (ny & (4096-1)) != 0){
    fprintf(stderr,"Error: %i is not a multiple of 4096.\n",(nx & (4096-1))!=0 ? nx : ny);
    exit(1);
  }

  fprintf(stdout,"Running jedidistort on image of shape (%li, %li).\nPixel scale: %f.\nLens redshift: %f.\n\n", nx, ny, scale, zl);

  /*parse lens list file*/
  if ((lensfile = fopen(argv[2], "r")) == NULL) {
    fprintf(stderr, "Error opening lens list file \"%s\".\n", argv[2]);
    exit(1);
  }
  fprintf(stdout, "Opened lens list file \"%s\".\n", argv[2]);

  //count the number of lenses
  nlenses = 0;
  while(fscanf(lensfile, "%f %f %i %f %f", &templens.x, &templens.y, &templens.type, &templens.p1, &templens.p2) > 0){
    nlenses++;
  }
  fprintf(stdout, "%li lenses in \"%s\".\n", nlenses, argv[2]);

  //initialize memory for all the galaxies
  lenses = (lens *) calloc(nlenses, sizeof(lens));
  if(lenses == NULL){
    fprintf(stderr,"Error allocating memory for lens list.");
    exit(1);
  }

  //cluster mass profile variables
  float       px_per_rad = 180.0*3600.0/(PI*scale); //conversion factor from radians to pixels
  float       rad_per_px = 1/px_per_rad;          //conversion factor from pixels to radians

  //read in the lenses
  rewind(lensfile);
  int nlens;
  for(nlens = 0; nlens < nlenses; nlens++){
    fscanf(lensfile, "%f %f %i %f %f", &lenses[nlens].x, &lenses[nlens].y, &lenses[nlens].type, &lenses[nlens].p1, &lenses[nlens].p2);
    fprintf(stdout,"Lens %i:\nx: %f\ny: %f\ntype: %i\np1: %f\np2: %f\n\n",nlens,lenses[nlens].x, lenses[nlens].y, lenses[nlens].type, lenses[nlens].p1, lenses[nlens].p2);

    //find the maximum radius we need to put in the table
    //we need the further corner from the lens

    float dx = (lenses[nlens].x > nx/2) ? lenses[nlens].x : nx-lenses[nlens].x;
    float dy = (lenses[nlens].y > ny/2) ? lenses[nlens].y : nx-lenses[nlens].y;
    float dr = sqrt(dx*dx + dy*dy);
    lenses[nlens].nr = (long int) ceil(dr*DR);   //the number of table entries is the max radius * the increment
    fprintf(stdout,"dx: %f\ndy: %f\ndr: %f\nnr: %li\n", dx, dy, dr, lenses[nlens].nr);

    //allocate space for the table
    lenses[nlens].table = (float *) calloc(lenses[nlens].nr, sizeof(float));
    if(lenses[nlens].table == NULL){
      fprintf(stderr,"Error allocating memory for lens %i table.\n", nlens);
    }

    //special case at zero
    switch (lenses[nlens].type){
    case 1: //Singular Isothermal Sphere profile
      {     
	long int     a; //counter in the table in units of [px/DR]
	lenses[nlens].table[0] = 0;     //special case at zero
	float alpha = (lenses[nlens].p1/C);
	alpha = px_per_rad*4*PI*alpha*alpha;
	for(a = 1; a < lenses[nlens].nr; a++){
	  lenses[nlens].table[a] = alpha*((float) DR)/((float) a);
	  //fprintf(stdout,"alpha: %f\n", alpha);
	}
      }
      break;
    case 2: //Navarro-Frenk-White profile
      {
	long int     a; //counter in the table in units of [px/DR]
	lenses[nlens].table[0] = 0;     //special case at zero
	//distance to the lens [Mpc]
	float   Dl = angular_di_dist(0, zl);
                    
	//calculate the concentration as a function of redshift zl
	//lenses[nlens].p2 = (6.71/pow((1+zl),0.44))*pow((0.678*lenses[nlens].p1*1E14/(2E12)),-0.091);
                    
	//rs parameter for converting from angle to unitless x variable
	//float   rs = cbrtf(G/(H0*H0))* (10.0/lenses[nlens].p2);
	//float   rs = 0.978146*cbrtf(lenses[nlens].p1)/lenses[nlens].p2;
	//calculate the Hubble constant H as a function of zl, then substitute into the equation of rs(z); this is a correction to the above equation
	float   rs = 0.978146*cbrtf(lenses[nlens].p1)* cbrtf(1/(OMEGA_M*(1+(1+zl)/OPZEQ)*(1+zl)*(1+zl)*(1+zl)+OMEGA_D))/ lenses[nlens].p2;
	
	float   x_per_rad = Dl/rs;
	//slightly modified version of NFW delta_c parameter
	float rdelta_c = 1/(logf(1+lenses[nlens].p2) - (lenses[nlens].p2/(1+lenses[nlens].p2)));

	//constant prefactor for bending angle
	//float NFW_prefactor = px_per_rad*4*G*lenses[nlens].p1*rdelta_c/(9*Dl*10000);
	float NFW_prefactor = px_per_rad*4*G*lenses[nlens].p1*rdelta_c/(9*Dl*1E5);


	fprintf(stdout,"\nDl: %f\nrs: %f\nrdelta_c: %f\ndelta_c: %f\nNFW_prefactor: %e\n\n", Dl, rs, rdelta_c, (200.0/3.0)*lenses[nlens].p2*lenses[nlens].p2*lenses[nlens].p2*rdelta_c, NFW_prefactor); 
	for(a = 1; a < lenses[nlens].nr; a++){
	  //convert to radians
	  double theta = (double) a *rad_per_px/((double) DR);
	  //convert to unitless x variable
	  double xvar = theta *( (double) x_per_rad);
	  //calculate the enclosed mass for this distance, up to a multiplicative constant
	  double Menc = log(xvar/2);
	  if(xvar > 0 && xvar < 1)
	    Menc += (2/sqrt(1-xvar*xvar))*atanh(sqrt((1-xvar)/(1+xvar)));
	  else if(xvar == 1)
	    Menc += 1;
	  else if(xvar > 1)
	    Menc += (2/sqrt(xvar*xvar-1))*atan(sqrt((xvar-1)/(1+xvar)));
	  else{
	    fprintf(stdout,"Error: could not calculate lens %i profile. xvar outside of range (0, infty). xvar: %f", nlens, xvar);
	    exit(1);
	  }

	  //put it all together;
	  double b = ((double) NFW_prefactor )*Menc*((double) DR)/(theta*a);
	  float b_f = (float) b;
	  lenses[nlens].table[a] = b_f;
	  //if((a % 10)==0)
	  //  fprintf(stdout,"table[%i]: alpha:%e\ttheta: %e\txvar: %e\tMenc: %e\n", a, b,theta, xvar, Menc);
	}
      }
      break;
    case 3: //Navarro-Frenk-White profile
      {

	//yes, this is hard-coded. since you can change the distance to center, overall mass doesn't really matter that much.
	float mass = 10;

	long int     a; //counter in the table in units of [px/DR]
	lenses[nlens].table[0] = 0;     //special case at zero
	/**********************************************************************test with SIS profile for grid simulation*/
	//distance to the lens [Mpc]
	float   Dl = angular_di_dist(0, zl);
                    
	//calculate the concentration as a function of redshift zl
	//lenses[nlens].p2 = (6.71/pow((1+zl),0.44))*pow((0.678*20*1E14/(2E12)),-0.091);
                    
	//rs parameter for converting from angle to unitless x variable
	//float   rs = cbrtf(G/(H0*H0))* (10.0/lenses[nlens].p2);
	float   rs = 0.978146*cbrtf(mass)/lenses[nlens].p2;
	float   x_per_rad = Dl/rs;
	//slightly modified version of NFW delta_c parameter
	float rdelta_c = 1/(logf(1+lenses[nlens].p2) - (lenses[nlens].p2/(1+lenses[nlens].p2)));

	//constant prefactor for bending angle
	float NFW_prefactor = px_per_rad*4*G*mass*rdelta_c/(9*Dl*1E5);


	fprintf(stdout,"\nDl: %f\nrs: %f\nrdelta_c: %f\ndelta_c: %f\nNFW_prefactor: %e\n\n", Dl, rs, rdelta_c, (200.0/3.0)*lenses[nlens].p2*lenses[nlens].p2*lenses[nlens].p2*rdelta_c, NFW_prefactor); 


	//convert to radians
	double theta = (double) lenses[nlens].p1 *rad_per_px;
	//convert to unitless x variable
	double xvar = theta *( (double) x_per_rad);
	//calculate the enclosed mass for this distance, up to a multiplicative constant
	double Menc = log(xvar/2);
	if(xvar > 0 && xvar < 1)
	  Menc += (2/sqrt(1-xvar*xvar))*atanh(sqrt((1-xvar)/(1+xvar)));
	else if(xvar == 1)
	  Menc += 1;
	else if(xvar > 1)
	  Menc += (2/sqrt(xvar*xvar-1))*atan(sqrt((xvar-1)/(1+xvar)));
	else{
	  fprintf(stdout,"Error: could not calculate lens %i profile. xvar outside of range (0, infty). xvar: %f", nlens, xvar);
	  exit(1);
	}

	for(a = 1; a < lenses[nlens].nr; a++){
	  //put it all together;
	  double b = ((double) NFW_prefactor )*Menc*((double) DR)/(theta*a);
	  float b_f = (float) b;
	  lenses[nlens].table[a] = b_f;
	  //if((a % 10)==0)
	  //fprintf(stdout,"%i\t%e\t%e\t%e\t%e\n", a, b,theta, xvar, Menc);
	}
	/***************************************************************************/
	/***************************************************************************
                    float alpha = (1000/C); //Use sigma_v=200km/s as an example
alpha = px_per_rad*4*PI*alpha*alpha;
for(a = 1; a < lenses[nlens].nr; a++){
lenses[nlens].table[a] = alpha*((float) DR)/((float) a);
//fprintf(stdout,"alpha: %f\n", alpha);
}
	***************************************************************************/
      }
      break;

    }
    fprintf(stdout,"Calculated profile for lens %i.\n\n", nlens);
  }
  fclose(lensfile);

  //make the grids
  int         gr;         //the current grid
  long int    row, col;   //row and column in the current grid
  long int    srow, scol; //row and column in the sub grid
  for(gr = 0; gr < ngr; gr++){
    //allocate memory for the ith grid
    ngx[gr] = nx >> b[gr];
    ngy[gr] = ny >> b[gr];
    fprintf(stdout,"making grid %i with sieve %i: (%i,%i).\n",gr, 1 << b[gr],ngx[gr], ngy[gr]);
    grids[gr] = (rect *) calloc(ngx[gr] * ngy[gr], sizeof(rect));
    if(grids[gr] == NULL){
      fprintf(stderr,"Error: could not allocate memory for grid array %i.", gr);
      exit(1);
    }
    /*The first grid is different because we need to compute alpha directly instead of drawing on the other grids.*/
    //loop over grid
    for(row = 0; row < ngx[gr]; row++){
      for(col = 0; col < ngy[gr]; col++){
	//loop over subgrid
	rect bounding_box;
	for(srow = 0; srow < g; srow++){
	  for(scol = 0; scol < g; scol++){
	    if(gr==0){
	      //get deflection angle
	      float alphax = 0, alphay = 0;
	      get_alpha((row*g+srow) << BNSUBPX, (col*g+scol) << BNSUBPX, nlenses, lenses, &alphax, &alphay);
	      //find maxima and minima
	      if(srow==0 && scol==0){
		bounding_box.xmax = alphax;
		bounding_box.xmin = alphax;
		bounding_box.ymax = alphay;
		bounding_box.ymin = alphay;
	      } else {
		if(alphay > bounding_box.ymax)
		  bounding_box.ymax = alphay;
		if(alphay < bounding_box.ymin)
		  bounding_box.ymin = alphay;

		if(alphax > bounding_box.xmax)
		  bounding_box.xmax = alphax;
		if(alphax < bounding_box.xmin)
		  bounding_box.xmin = alphax;
	      }
	    } else {
	      //get index in finer array
	      int index = (ngy[gr-1]*(col*g+scol))+row*g+srow;                            
	      /*rect finer_bb = {grids[gr-1][index].xmin, grids[gr-1][index].xmax, grids[gr-1][index].ymin, grids[gr-1][index].ymax};*/
	      rect finer_bb;
	      finer_bb.xmin = grids[gr-1][index].xmin;
	      finer_bb.xmax = grids[gr-1][index].xmax;
	      finer_bb.ymin = grids[gr-1][index].ymin;
	      finer_bb.ymax = grids[gr-1][index].ymax;
	      //find maxima and minima 
	      if(srow ==0 && scol==0){
		bounding_box.xmax = finer_bb.xmin;
		bounding_box.xmin = finer_bb.xmax;
		bounding_box.ymax = finer_bb.ymax;
		bounding_box.ymin = finer_bb.ymin;
	      } else {
		if(finer_bb.xmax > bounding_box.xmax)
		  bounding_box.xmax = finer_bb.xmax;
		if(finer_bb.xmin < bounding_box.xmin)
		  bounding_box.xmin = finer_bb.xmin;
		if(finer_bb.ymax > bounding_box.ymax)
		  bounding_box.ymax = finer_bb.ymax;
		if(finer_bb.ymin < bounding_box.ymin)
		  bounding_box.ymin = finer_bb.ymin;
	      }
	    }
	  }
	}
	grids[gr][ngy[gr]*col + row].ymax = bounding_box.ymax;
	grids[gr][ngy[gr]*col + row].ymin = bounding_box.ymin;
	grids[gr][ngy[gr]*col + row].xmax = bounding_box.xmax;
	grids[gr][ngy[gr]*col + row].xmin = bounding_box.xmin;

	/*if(gr > 1){
	    fprintf(stdout,"gr: %i, (%i, %i)", gr, row, col);
	      print_rect(&grids[gr][ngy[gr]*col+row]);
	      }*/
      }
    }
  }
}

void setupPaste() {
	fimage = (float *) calloc(nx*ny, sizeof(float)); //make final image
	if (fimage == NULL) {
		fprintf(stderr, "Error: could not allocate memory for the final image.");
		exit(1);
	}
	//make the mutex array (These will prevent the threads from trying to pull from the same file at once)
	bandlocks = calloc(NUMBANDS, sizeof(pthread_mutex_t));
	if (bandlocks == NULL) {
		fprintf(stderr, "Error: could not allocate memory for mutex locks");
		exit(1);
	}

	//initialize mutex array used to look input files
	int g;
	for(g = 0; g < NUMBANDS; g++) {
		if (pthread_mutex_init(&bandlocks[g], NULL) != 0) {
			fprintf(stderr, "Error: mutex init has failed\n");
			exit(1);
		}
	}
}

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#define min(x, y)               ((x <= y) ? x : y)
#define max(x, y)	        ((x >= y) ? x : y)
#define prob_push_right(s)      (1.0 / (1.0 + exp(-max(-50.0, min(s, 50.0)))))
#define random                  ((float) rand() / (float)((1 << 31)-1))

#define N_BOXES         162         /* Number of disjoint boxes of state space. */
#define ALPHA		1000        /* Learning rate for action weights, w. */
#define BETA		0.5         /* Learning rate for critic weights, v. */
#define GAMMA		0.95        /* Discount factor for critic. */
#define LAMBDAw		0.9         /* Decay rate for w eligibility trace. */
#define LAMBDAv		0.8         /* Decay rate for v eligibility trace. */

#define MAX_FAILURES     100         /* Termination criterion. */
#define MAX_STEPS        100000

/*----------------------------------------------------------------------
   cart_pole:  Takes an action (0 or 1) and the current values of the
 four state variables and updates their values by estimating the state
 TAU seconds later.
----------------------------------------------------------------------*/

/*** Parameters for simulation ***/

#define GRAVITY 9.8
#define MASSCART 1.0
#define MASSPOLE 0.1
#define TOTAL_MASS (MASSPOLE + MASSCART)
#define LENGTH 0.5      /* actually half the pole's length */
#define POLEMASS_LENGTH (MASSPOLE * LENGTH)
#define FORCE_MAG 10.0
#define TAU 0.02      /* seconds between state updates */
#define FOURTHIRDS 1.3333333333333


void cart_pole(action, x, x_dot, theta, theta_dot)
int action;
float *x, *x_dot, *theta, *theta_dot;
{
    float xacc,thetaacc,force,costheta,sintheta,temp;

    force = (action>0)? FORCE_MAG : -FORCE_MAG;
    costheta = cos(*theta);
    sintheta = sin(*theta);

    temp = (force + POLEMASS_LENGTH * *theta_dot * *theta_dot * sintheta)
             / TOTAL_MASS;

    thetaacc = (GRAVITY * sintheta - costheta* temp)
         / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta
                                              / TOTAL_MASS));

    xacc  = temp - POLEMASS_LENGTH * thetaacc* costheta / TOTAL_MASS;

/*** Update the four state variables, using Euler's method. ***/

    *x  += TAU * *x_dot;
    *x_dot += TAU * xacc;
    *theta += TAU * *theta_dot;
    *theta_dot += TAU * thetaacc;
}

/*----------------------------------------------------------------------
   get_box:  Given the current state, returns a number from 1 to 162
  designating the region of the state space encompassing the current state.
  Returns a value of -1 if a failure state is encountered.
----------------------------------------------------------------------*/

#define one_degree 0.0174532  /* 2pi/360 */
#define six_degrees 0.1047192
#define twelve_degrees 0.2094384
#define fifty_degrees 0.87266

int get_box(x,x_dot,theta,theta_dot)
float x,x_dot,theta,theta_dot;
{
  int box=0;

  if (x < -2.4 ||
      x > 2.4  ||
      theta < -twelve_degrees ||
      theta > twelve_degrees)          return(-1); /* to signal failure */

  if (x < -0.8)            box = 0;
  else if (x < 0.8)              box = 1;
  else                         box = 2;

  if (x_dot < -0.5)            ;
  else if (x_dot < 0.5)                box += 3;
  else                     box += 6;

  if (theta < -six_degrees)          ;
  else if (theta < -one_degree)        box += 9;
  else if (theta < 0)            box += 18;
  else if (theta < one_degree)         box += 27;
  else if (theta < six_degrees)        box += 36;
  else                   box += 45;

  if (theta_dot < -fifty_degrees)   ;
  else if (theta_dot < fifty_degrees)  box += 54;
  else                                 box += 108;

  return(box);
}

typedef float vector[N_BOXES];

int main()
{
  float x,			/* cart position, meters */
        x_dot,			/* cart velocity */
        theta,			/* pole angle, radians */
        theta_dot;		/* pole angular velocity */
  vector  w,			/* vector of action weights */
          v,			/* vector of critic weights */
          e,			/* vector of action weight eligibilities */
          xbar;			/* vector of critic weight eligibilities */
  float p, oldp, rhat, r;
  int box, i, steps = 0, failures=0, failed; bool y;

  srand( time(NULL) );

  /*--- Initialize action and heuristic critic weights and traces. ---*/
  for (i = 0; i < N_BOXES; i++)
    w[i] = v[i] = xbar[i] = e[i] = 0.0;

  /*--- Starting state is (0 0 0 0) ---*/
  x = x_dot = theta = theta_dot = 0.0;

  /*--- Find box in state space containing start state ---*/
  box = get_box(x, x_dot, theta, theta_dot);

  /*--- Iterate through the action-learn loop. ---*/
  while (steps++ < MAX_STEPS && failures < MAX_FAILURES)
    {
      /*--- Choose action randomly, biased by current weight. ---*/
      y = (bool)(random < prob_push_right(w[box]));

      /*--- Update traces. ---*/
      e[box] += (1.0 - LAMBDAw) * ((int)y - 0.5);
      xbar[box] += (1.0 - LAMBDAv);

      /*--- Remember prediction of failure for current state ---*/
      oldp = v[box];

      /*--- Apply action to the simulated cart-pole ---*/
      cart_pole((int)y, &x, &x_dot, &theta, &theta_dot);

      /*--- Get box of state space containing the resulting state. ---*/
      box = get_box(x, x_dot, theta, theta_dot);

      if (box < 0)		
	{
	  /*--- Failure occurred. ---*/
	  failed = 1;
	  failures++;
	  printf("Trial %d was %d steps.\n", failures, steps);
	  steps = 0;

	  /*--- Reset state to (0 0 0 0).  Find the box. ---*/
	  x = x_dot = theta = theta_dot = 0.0;
	  box = get_box(x, x_dot, theta, theta_dot);

	  /*--- Reinforcement upon failure is -1. Prediction of failure is 0. ---*/
	  r = -1.0;
	  p = 0.;
	}
      else
	{
 	  /*--- Not a failure. ---*/
	  failed = 0;

	  /*--- Reinforcement is 0. Prediction of failure given by v weight. ---*/
	  r = 0;
	  p= v[box];
	}

      /*--- Heuristic reinforcement is:   current reinforcement
	      + gamma * new failure prediction - previous failure prediction ---*/
      rhat = r + GAMMA * p - oldp;

      for (i = 0; i < N_BOXES; i++)
	{
	  /*--- Update all weights. ---*/
	  w[i] += ALPHA * rhat * e[i];
	  v[i] += BETA * rhat * xbar[i];
	  if (v[i] < -1.0)
	    v[i] = v[i];

	  if (failed)
	    {
	      /*--- If failure, zero all traces. ---*/
	      e[i] = 0.;
	      xbar[i] = 0.;
	    }
	  else
	    {
	      /*--- Otherwise, update (decay) the traces. ---*/	      
	      e[i] *= LAMBDAw;
	      xbar[i] *= LAMBDAv;
	    }
	}

    }
  if (failures == MAX_FAILURES)
    printf("Pole not balanced. Stopping after %d failures.",failures);
  else
    printf("Pole balanced successfully for at least %d steps\n", steps);
  return(0);
}



package gradDescent;

import gradDescent.Expression.*;

public class Main {
	
	// convert an array of objects to a string representation
	public static String vectorToString(Object[] v) {
		String s = "[";
		for (int i = 0; i < v.length-1; i++) {
			s = s.concat(v[i].toString()).concat(", ");
		}
		s = s.concat(v[v.length-1].toString()).concat("]");
		return s;
	}

	// convert a string to an expression
	// variables are denoted x_0, x_1, etc.
	public static Expr readLine(String line) {
		// remove spaces
		line = line.replace(" ", "");
		// check if the expression is in brackets and remove
		if (line.charAt(0) == '(' && line.charAt(line.length() - 1) == ')') {
			// read right to left, ignore first and last characters
			int bLevel = 0; // number of brackets
			boolean inBrackets = true;
			for (int i = line.length() - 2; i > 0; i--) {
				if (line.charAt(i) == ')') bLevel++;
				if (line.charAt(i) == '(') bLevel--;
				if (bLevel < 0) {
					inBrackets = false;
					break;
				}
			}
			if (inBrackets) return readLine(line.substring(1, line.length() - 1));
		}
		// check for operations not in brackets
		// use recursion so look for last operation to be performed
		int bLevel = 0;
		int addIndex = -1; // index of last addition
		int mulIndex = -1; // index of last multiplication
		int powIndex = -1; // index of last exponent
		for (int i = 0; i < line.length(); i++) {
			if (line.charAt(i) == ')') bLevel--;
			if (line.charAt(i) == '(') bLevel++;
			if (line.charAt(i) == '+' && bLevel == 0) addIndex = i;
			if (line.charAt(i) == '*' && bLevel == 0) mulIndex = i;
			if (line.charAt(i) == '^' && bLevel == 0) powIndex = i;
		}
		// if addition found do it
		if (addIndex > -1) {
			return new Add (
					readLine(line.substring(0, addIndex)),
					readLine(line.substring(addIndex + 1, line.length()))
				);
		}
		// if multiplication found do it
		if (mulIndex > -1) {
			return new Mul (
					readLine(line.substring(0, mulIndex)),
					readLine(line.substring(mulIndex + 1, line.length()))
				);
		}
		// if exponent found do it
		if (powIndex > -1) {
			return new Pow (
					readLine(line.substring(0, powIndex)),
					Double.parseDouble(line.substring(powIndex + 1, line.length()))
				);
		}
		// if line is 'x_n', return id(n)
		if (line.length() > 2 && line.substring(0, 2).contentEquals("x_")) {
			int n = Integer.parseInt(line.substring(2));
			return new Id(n);
		}
		// otherwise return constant
		return new Const(Double.parseDouble(line));
	}
	
	// gradient descent to find minimum of function
	// start at w and take steps in the direction of the gradient
	// length of step is norm of gradient times alpha
	public static double[] gradDescent(Expr g, double[] w, int its, double alpha) {
		Expr[] grad = g.gradient();
		double[] r = w.clone();

		double[] gradVal = new double[r.length];
		for (int i = 0; i < its; i++) {
			// calculate gradient at current point
			for (int j = 0; j < r.length; j++) {
				gradVal[j] = grad[j].evaluate(r);
			}
			// calculate new point
			for (int j = 0; j < r.length; j++) {
				r[j] = r[j] - gradVal[j] * alpha;
			}
		}
		
		return r;
	}
	
	// normed gradient descent
	// length of step is alpha times 1/j where j is the step number
	// useful for flat valleys where gradient is very small
	public static double[] normedGradDescent(Expr g, double[] w, int its, double alpha, double eps) {
		Expr[] grad = g.gradient();
		double[] r = w.clone();

		double[] gradVal = new double[r.length];
		for (int i = 0; i < its; i++) {
			// calculate gradient and norm at current point
			double gradNorm = 0;
			for (int j = 0; j < r.length; j++) {
				gradVal[j] = grad[j].evaluate(r);
				gradNorm = gradNorm + Math.pow(gradVal[j], 2);
			}
			gradNorm = Math.pow(gradNorm, 0.5) + eps;
			// calculate new point, use 1/j
			for (int j = 0; j < r.length; j++) {
				r[j] = r[j] - gradVal[j] / gradNorm * alpha;
			}
		}
		
		return r;
	}
	
	// Expr for least squared cost function with linear regression
	public static Expr linearRegressionCost(double[][] xVals, double[] yVals) {
		Expr cost = null;
		for (int i = 0; i < yVals.length; i++) {
			// new term will look like (w_0 + w_1 * x_i,1 + ...  + w_n * x_i,n - y_i)^2
			// w_j are the variables in the result expression
			Expr newTerm = new Id(0); // w_0
			for (int j = 0; j < xVals[0].length; j++) {
				newTerm = new Add(newTerm, new Mul(new Id(j+1), new Const(xVals[i][j]))); // w_j * x_i,j
			}
			newTerm = new Add(newTerm, new Const(-yVals[i])); // -y_i
			newTerm = new Pow(newTerm, 2);
			if (cost == null) cost = newTerm;
			else cost = new Add(cost, newTerm);
		}
		cost = new Mul(new Const(1.0 / yVals.length), cost);
		return cost;
	}
	
	
	public static void main(String[] args) {
		Const e1 = new Const(2);
		Mul e2 = new Mul(new Const(3), new Id(1));
		Add e3 = new Add(e1, e2); // expression 2 + 3*x1
		
		Expr e4 = readLine("x_0^2 + (x_1+(-2))^4");
		
		double[] x = {2, 3};
		
		// print expressions and gradients
		System.out.println(e3.toString());
		System.out.println(vectorToString(e3.gradient()));
		System.out.println(e4.toString());
		System.out.println(vectorToString(e4.gradient()));
		
		// find minimum of e4
		double[] w = normedGradDescent(e4, x, 10000, 1, 1E-6);
		double[] w2 = gradDescent(e4, x, 1000, 0.01);
		System.out.printf("[%f, %f]\n", w[0], w[1]);
		System.out.printf("[%f, %f]\n", w2[0], w2[1]);
		
		// linear regression
		
		// generate some test data around the plane 3x0 + 2x1 - 2
		int dataSize = 1000;
		double[][] xs = new double[dataSize][2];
		double[] y1s = new double[dataSize];
		double[] y2s = new double[dataSize];
		
		for (int i = 0; i < dataSize; i++) {
			xs[i][0] = Math.random() * 10; // xs are vectors in [0, 10) x [0, 10)
			xs[i][1] = Math.random() * 10; 
			y1s[i] = 3*(xs[i][0]) + 2*(xs[i][1]) - 2; // exactly on the line
			y2s[i] = y1s[i] + Math.random() - 0.5; // add noise
		}
		
		// do linear regression on y1s
		Expr cost = (linearRegressionCost(xs, y1s));
		
		double[] start = new double[3]; // starting point (0, 0, 0) for gradient descent
		
		double[] weights = gradDescent(cost, start, 10000, 0.01);
		System.out.println(cost.evaluate(weights)); // final cost, should be near 0
		System.out.printf("[%f, %f, %f]\n", weights[0], weights[1], weights[2]); // expect [-2, 3, 2]
		
		// do linear regression on y2s
		Expr cost2 = (linearRegressionCost(xs, y2s));
	
		double[] weights2 = gradDescent(cost2, start, 10000, 0.01);
		System.out.println(cost2.evaluate(weights2)); // final cost, will be greater than 0
		System.out.printf("[%f, %f, %f]\n", weights2[0], weights2[1], weights2[2]); // expect near (but not exactly) [-2, 3, 2]
		
		

	}

}

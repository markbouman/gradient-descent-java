package gradDescent;

import java.lang.Math;

// expressions which depend on x_0, x_1, etc.
public class Expression {
	
	// abstract class defining evaluation and derivative
	protected static abstract class Expr {
		protected int nVars; // number of variables in the expression
		
		// getter for nVars
		public int nVars() {
			return nVars;
		}
		
		// convert the expression to a string
		public abstract String toString();
		
		// evaluate the expression at x
		public abstract double evaluate(double[] x);
		
		// return an expression for the derivative
		public abstract Expr derivative(int var);
		
		// simplify the expression
		public abstract Expr simplify();
		
		// return an array of expressions for the gradient (which is a vector)
		public Expr[] gradient() {
			Expr[] grad = new Expr[nVars];
			for (int i = 0; i < nVars; i++) {
				grad[i] = derivative(i);
			}
			return grad;
		}
	}
	
	// constant expression
	public static class Const extends Expr {
		private double val;
		
		public double getVal() {
			return val;
		}
		
		public Const(double new_val) {
			nVars = 0;
			val = new_val;
		}
		
		public String toString() {
			return String.valueOf(val);
		}
		
		public double evaluate(double[] x) {
			return val;
		}
		
		public Expr derivative(int x) {
			return (new Const(0));
		}
		
		public Expr simplify() {
			return this;
		}
	}
	
	// identity expression for x_n
	public static class Id extends Expr {
		private int n; // variable number
		
		public Id(int new_n) {
			nVars = new_n + 1;
			n = new_n;
		}
		
		public int varNum() {
			return n;
		}
		
		public String toString() {
			return String.format("x_%d", n);
		}
		
		public double evaluate(double[] x) {
			return x[n];
		}
		
		public Expr derivative(int x) {
			if (x == n) return (new Const(1));
			else return (new Const(0));
		}
		
		public Expr simplify() {
			return this;
		}
	}
	
	// addition
	public static class Add extends Expr {
		private Expr a;
		private Expr b;
		
		public String toString() {
			return String.format("%s + %s", a.toString(), b.toString());
		}
		
		public Add(Expr new_a, Expr new_b) {
			nVars = Math.max(new_a.nVars, new_b.nVars);
			a = new_a;
			b = new_b;
		}
		
		public double evaluate(double[] x) {
			return (a.evaluate(x) + b.evaluate(x));
		}
		
		public Expr derivative(int n) {
			return (new Add(a.derivative(n), b.derivative(n))).simplify();
		}
		
		public Expr simplify() {
			if (a.getClass() == Const.class && ((Const) a).getVal() == 0) return b;
			if (b.getClass() == Const.class && ((Const) b).getVal() == 0) return a;
			if (a.getClass() == Const.class && a.getClass() == Const.class) {
				double aVal = ((Const) a).getVal();
				double bVal = ((Const) b).getVal();
				return new Const(aVal + bVal);
			}
			return this;
		}
	}
	
	// multiplication
	public static class Mul extends Expr {
		private Expr a;
		private Expr b;
		
		public Mul(Expr new_a, Expr new_b) {
			nVars = Math.max(new_a.nVars, new_b.nVars);
			a = new_a;
			b = new_b;
		}
		
		public String toString() {
			String astr = a.toString();
			String bstr = b.toString();
			if (a.getClass() == Add.class) astr = String.format("(%s)", astr);
			if (b.getClass() == Add.class) bstr = String.format("(%s)", bstr);
			return String.format("%s*%s", astr, bstr);
		}
		
		public double evaluate(double[] x) {
			return (a.evaluate(x) * b.evaluate(x));
		}
		
		// product rule for derivative
		public Expr derivative(int n) {
			return (new Add(
					(new Mul (a.derivative(n), b)).simplify(),
					(new Mul (a, b.derivative(n))).simplify()
				)).simplify();
		}
		
		public Expr simplify() {
			if (a.getClass() == Const.class && ((Const) a).getVal() == 0) return (new Const(0));
			if (b.getClass() == Const.class && ((Const) b).getVal() == 0) return (new Const(0));
			if (a.getClass() == Const.class && ((Const) a).getVal() == 1) return b;
			if (b.getClass() == Const.class && ((Const) b).getVal() == 1) return a;
			if (a.getClass() == Const.class && b.getClass() == Const.class) {
				double aVal = ((Const) a).getVal();
				double bVal = ((Const) b).getVal();
				return new Const(aVal * bVal);
			}
			return this;
		}
	}
	
	// polynomial term b^p, constant power
	public static class Pow extends Expr {
		private Expr b;
		private double p;
		
		public Pow(Expr new_b, double new_p) {
			nVars = new_b.nVars;
			b = new_b;
			p = new_p;
		}
		
		public String toString() {
			String bstr = b.toString();
			if (b.getClass() == Add.class || b.getClass() == Mul.class) bstr = String.format("(%s)", bstr);
			return String.format("%s^%f", bstr, p);			
		}
		
		public double evaluate(double[] x) {
			return Math.pow(b.evaluate(x), p);
		}
		
		// chain rule
		public Expr derivative(int n) {
			return new Mul (
					new Mul(new Const(p), new Pow(b, p-1).simplify()).simplify(),
					b.derivative(n)
				).simplify();
		}
		
		public Expr simplify() {
			if (p == 0) return (new Const(1));
			if (p == 1) return b;
			if (b.getClass() == Const.class) return new Const(Math.pow(((Const) b).getVal(), p));
			return this;
		}
	}
	
}
import sys

import numpy as np
import sympy
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLineEdit, QLabel
from PyQt5.QtCore import QTimer, QUrl,QEventLoop
from PyQt5.QtMultimedia import QSound, QMediaPlayer, QMediaContent
from PyQt5 import QtGui, QtCore
from gui import Ui_MainWindow2
import copy
import math
from mpmath import mp, matrix, mpf,sqrt
# from Equation import Expression
import time
import sys
import matplotlib.pyplot as plt
from sympy import symbols, sympify, diff







class Solver(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow2()
        self.ui.setupUi(self)
        self.setWindowTitle("Solver")
        self.singleStep=False
        self.sig=self.ui.figures.value()
        self.methodSelect(0)
        self.Eps = float(self.ui.Error.text())
        self.numit=50
        self.x0=0
        self.x1=0
        self.xl=0
        self.xu=0
        self.m=1
        self.loop=QEventLoop()
        self.ui.pushButton_2.hide()
        self.equation=None
        self.aux=None
        self.ui.method.currentIndexChanged.connect(self.methodSelect)
        self.ui.singlestep.stateChanged.connect(self.checkbox)
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.update2d(self.ui.maingrid,0,None,True,False))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.update2d(self.lu.lu.Lgrid,1,None,True,False))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.update2d(self.lu.lu.Ugrid,1,None,True,False))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.ui.bgrid,0,None))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.updateconstants(self.ui.unknowngrid))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.updateconstants(self.ui.solutionconstgrid))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.ui.solutiongrid,1,None))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.updateconstants(self.iterative.iterative.iterativeconst))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.updateconstants(self.lu.lu.solutionconstgrid))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.iterative.iterative.guessgrid,0,None))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.iterative.iterative.iterativesol,1,None))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.lu.lu.solutiongrid,1,None))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.gausswindow.gauss.solutiongrid,1,None))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.updateconstants(self.gausswindow.gauss.solutionconstgrid))
        # self.ui.numunknown.currentIndexChanged.connect(lambda :self.update2d(self.gausswindow.gauss.auggrid,1,None,True,True))



        self.ui.pushButton_2.clicked.connect(self.loop.quit)
        self.ui.Plot.clicked.connect(self.plot_function)
        self.ui.figures.valueChanged.connect(self.figures)
        self.ui.start.clicked.connect(self.start)
        self.ui.back.clicked.connect(self.tohome)

    def tohome(self):
        from mainback import Home

        self.home=Home()
        self.home.show()
        self.close()
    def start(self):
        self.Eps = float(self.ui.Error.text())
        self.x0 = float(self.ui.x0.text())
        self.x1 = float(self.ui.x1.text())
        self.xl = float(self.ui.xl.text())
        self.xu = float(self.ui.xu.text())
        self.m = int(self.ui.lineEdit.text())
        self.equation=self.ui.mainfunction.text()
        self.ui.sol.setText('')
        self.ui.totalit.setText('')
        self.ui.time.setText('')
        self.ui.message.setText('')
        self.ui.xlit.setText('')
        self.ui.xu_2.setText('')
        self.ui.x0it.setText('')
        self.ui.x1it.setText('')
        self.ui.message.setText('')

        self.numit=self.ui.numit.value()
        print('here')
        if self.singleStep==True:
            self.ui.pushButton_2.show()
        if self.ui.method.currentIndex()==0:
            start = time.time()
            self.bisection(self.xl,self.xu,self.equation,self.Eps,self.numit,self.sig)
        if self.ui.method.currentIndex()==1:
            start = time.time()
            self.falsePosition(self.equation,self.xl,self.xu,self.numit,self.Eps,self.sig)
        if self.ui.method.currentIndex() == 2:
            self.aux=self.ui.gx.text()
            start = time.time()
            self.fixed_point_iteration(self.equation,self.aux,self.x0,self.sig,self.Eps,self.numit)
        if self.ui.method.currentIndex()==3:
            start = time.time()
            self.newton(self.equation,self.numit,self.Eps,self.x0,self.sig)
        if self.ui.method.currentIndex()==4:
            start = time.time()
            self.modifiednewton1(self.equation,self.numit,self.Eps,self.x0,self.m,self.sig)
        if self.ui.method.currentIndex()==5:
            start = time.time()
            self.modifiednewton2(self.equation,self.numit,self.Eps,self.x0,self.sig)
        if self.ui.method.currentIndex()==6:
            start = time.time()
            self.secant_method_1(self.equation,self.x0,self.x1,self.Eps,self.numit,self.sig)
        end=time.time()

        self.ui.time.setText(str(round(end-start,10)))
    def methodSelect(self,index):
        self.ui.xl.setDisabled(True)
        self.ui.xu.setDisabled(True)
        self.ui.x0.setDisabled(True)
        self.ui.x1.setDisabled(True)
        self.ui.gx.setDisabled(True)
        self.ui.lineEdit.setDisabled(True)
        self.ui.sol.setText('')
        self.ui.totalit.setText('')
        self.ui.time.setText('')
        self.ui.message.setText('')

        if index==2:
            self.ui.gx.setDisabled(False)
        if index==4:
            self.ui.lineEdit.setDisabled(False)

        if index==0 or index==1 :
            self.ui.xl.setEnabled(True)
            self.ui.xu.setEnabled(True)

        elif index==2 or index==3 or index==4 or index==5:
            self.ui.x0.setEnabled(True)
        elif index==6:
            self.ui.x0.setEnabled(True)
            self.ui.x1.setEnabled(True)
    def checkbox(self):
        self.singleStep=not self.singleStep
    def figures(self):
        self.sig =self.ui.figures.value()
        print(self.sig)
    def plot_function(self):
        self.equation=self.ui.mainfunction.text()
        x = symbols('x')  # Define x as a symbolic variable
        expr = sympify(self.equation)  # Convert the string expression to a sympy expression

        # Generate y values for the plot by substituting x in the expression
        x_vals = [i * 0.1 for i in range(-100, 101)]  # Generating x values from -10 to 10
        y_vals = [expr.subs(x, x_val) for x_val in x_vals]
        ax = plt.gca()
        ax.spines['left'].set_position('zero')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_position('zero')
        ax.spines['bottom'].set_color('black')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # Plotting the function
        plt.plot(x_vals, y_vals)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Plot of the Function')
        plt.show()
    def bisection(self,xl, xu, expression, epsilon=0.00001, max_iterations=50, significant_figures=20):
        fn = sympy.sympify(expression)
        tolerance = epsilon
        mp.dps = significant_figures
        xu = mp.mpf(xu)
        xl = mp.mpf(xl)
        if(self.singleStep==True):
            self.ui.xu_2.setText(str(xu))
            self.ui.xlit.setText(str(xl))
            self.loop.exec_()
        counter_for_number_of_iterations = 0
        while counter_for_number_of_iterations < max_iterations:
            # save previous value for xr to calculate error
            if counter_for_number_of_iterations > 0:
                xr_previous = xr
            # update current value for xr
            xr = mp.mpf((xl + xu) / 2.0)
            if counter_for_number_of_iterations > 0:
                error = mp.mpf(math.fabs(mp.mpf((xr - xr_previous) / xr)))
                if error < epsilon:
                    self.ui.totalit.setText(str(counter_for_number_of_iterations))
                    self.ui.sol.setText(str(xr))
                    self.ui.pushButton_2.hide()
                    return xr
            counter_for_number_of_iterations += 1
            evaluated_fxr = fn.evalf(mp.dps, subs={'x': xr})
            evaluated_fxu = fn.evalf(mp.dps, subs={'x': xu})
            evaluated_fxl = fn.evalf(mp.dps, subs={'x': xl})


            if evaluated_fxl * evaluated_fxu >= 0:
                self.ui.pushButton_2.hide()
                self.ui.message.setText("Initial bounds do not bracket the root.")
                return "Initial bounds do not bracket the root."

            elif evaluated_fxr == 0:
                self.ui.totalit.setText(str(counter_for_number_of_iterations))
                self.ui.sol.setText(str(mp.mpf(xr)))
                self.ui.pushButton_2.hide()
                return xr  # Exact root



            elif evaluated_fxl * evaluated_fxr < 0:
                xu = mp.mpf(xr)
                if (self.singleStep == True):
                    self.ui.xu_2.setText(str(xu))
                    self.ui.xlit.setText(str(xl))
                    self.loop.exec_()
            else:
                xl = mp.mpf(xr)
                if (self.singleStep == True):
                    self.ui.xu_2.setText(str(xu))
                    self.ui.xlit.setText(str(xl))
                    self.loop.exec_()
        if counter_for_number_of_iterations == 50:
            self.ui.pushButton_2.hide()
            self.ui.message.setText("Reached max iterations can't converge for the given data")
            return "Reached max iterations can't converge for the given data"

        # return mp.mpf(xr)
        print(xr)
        self.ui.pushButton_2.hide()
        self.ui.totalit.setText(str(counter_for_number_of_iterations))
        self.ui.sol.setText(str(mp.mpf(xr)))


    def secant_method_1(self,expression, x0, x1, tolerance=1e-4, max_iterations=1000, sig_figs=3):
        f = sympy.sympify(expression)
        mp.dps = sig_figs
        x0 = mp.mpf(x0)
        x1 = mp.mpf(x1)
        if (self.singleStep == True):
            self.ui.x0it.setText(str(x0))
            self.ui.x1it.setText(str(x1))
            self.loop.exec_()
        tolerance = mp.mpf(tolerance)
        divergence_threshold = mp.mpf(1e3)  # A large value indicating potential divergence
        previous_diff = mp.mpf('inf')  # Initialize to infinity
        #sign_change_count = 0  # Counter for consecutive sign changes
        print(max_iterations)
        for i in range(max_iterations):
            fx0 = mp.mpf(f.subs('x', x0).evalf())
            fx1 = mp.mpf(f.subs('x', x1).evalf())

            # Check for division by zero or very small denominator
            if mp.fabs(fx1 - fx0) < mp.power(10, -sig_figs):
                print('lol')
                break

            # Calculate the new approximation
            x_new = x1 - fx1 * (x1 - x0) / (fx1 - fx0)

            # Check for convergence
            if mp.fabs(x_new - x1) < tolerance:
                self.ui.sol.setText(str(mp.nstr(x_new, sig_figs)))
                self.ui.totalit.setText(str(i))
                self.ui.pushButton_2.hide()
                return str(mp.nstr(x_new, sig_figs))

            # Check for divergence
            if mp.fabs(x_new) > divergence_threshold:
                print("Diverging",x_new)
                self.ui.pushButton_2.hide()
                self.ui.message.setText("The method is diverging.")
                raise RuntimeError("The method is diverging.")

                # Update values for next iteration
            x0, x1 = x1, x_new
            if (self.singleStep == True):
                self.ui.x0it.setText(str(x0))
                self.ui.x1it.setText(str(x1))
                self.loop.exec_()
                # previous_diff = current_diff
        self.ui.pushButton_2.hide()
        self.ui.sol.setText(str(mp.nstr(x_new, sig_figs)))
        self.ui.totalit.setText(str(i))
        return str(mp.nstr(x1,sig_figs))


    def fixed_point_iteration(self,f_function, g_function, initial_guess, significant_figures, tolerance=1e-6,    max_iterations=100):
        mp.dps = significant_figures   # Adjust dps to include an extra digit for rounding
        try:
            # Parse the user-input functions
            f = lambda x: mpf(eval(f_function, {'x': x, 'sqrt': sqrt, 'e': math.e}))
            g = lambda x: mpf(eval(g_function, {'x': x, 'sqrt': sqrt, 'e': math.e}))

            x = mpf(initial_guess)
            if(self.singleStep==True):
                self.ui.x0it.setText(str(x))
                self.loop.exec_()
            iteration = 0

            while iteration < max_iterations:
                x_next = g(x)
                # Check for convergence
                if abs((x_next - x) / x_next) < tolerance:
                    print(f"The number of iterations: {iteration}")
                    self.ui.pushButton_2.hide()
                    self.ui.totalit.setText(str(iteration))
                    self.ui.sol.setText(str(x_next))
                    return x_next

                x = x_next
                if (self.singleStep == True):
                    self.ui.x0it.setText(str(x))
                    self.loop.exec_()
                iteration += 1

            # If max_iterations is reached without convergence
            self.ui.pushButton_2.hide()
            self.ui.message.setText("Fixed-point iteration did not converge within the specified number of iterations.")
            # raise RuntimeError("Fixed-point iteration did not converge within the specified number of iterations.")

        except Exception as e:
            self.ui.pushButton_2.hide()
            self.ui.message.setText(f"Error in function evaluation: {e}")
            # raise RuntimeError(f"Error in function evaluation: {e}")


    def newton(self,StringEquation, iterations, epsilon, initialguess, significantfigures=3):
        mp.dps = significantfigures
        flag = 0
        print("entered")
        x0 = mp.mpf(initialguess)
        relerror = mp.mpf(0.0)
        xprev = mp.mpf(0.0)
        deriv = sympify(diff(StringEquation))
        Equation = sympify(StringEquation)
        counter=0
        if(self.singleStep == True):
            self.ui.x0it.setText(str(x0))
            self.loop.exec_()

        for i in range(0, iterations):

            counter=i
            xprev = mp.mpf(x0)

            fx0 = Equation.evalf(mp.dps, subs={'x': x0})

            fx0_dash = deriv.evalf(mp.dps, subs={'x': x0})

            if (fx0_dash == 0):

                print("Error division by zero\n")
                self.ui.pushButton_2.hide()

                break

            else:

                x1 = mp.mpf((x0 - (fx0 / fx0_dash)))
                x2 = mp.mpf(x1)
                relerror = mp.mpf((math.fabs((x1 - xprev) / x1)))
                print("relerror", relerror)
                print(f"number={i}"  f"   x0=  {x0}" f"   F(X)=  {fx0}" f"   F(X)`=  {fx0_dash}"  f"     x1=  {x1}")

            if (abs(Equation.subs("x", x1)) == 0):
                xprev = mp.mpf(x0)
                x0 = mp.mpf(x1)
                flag = 1
                self.ui.pushButton_2.hide()

                break
            else:
                if (self.singleStep == True):
                    self.ui.x0it.setText(str(x0))
                xprev = mp.mpf(x0)
                x0 = mp.mpf(x1)
                if (self.singleStep == True):
                    self.ui.x0it.setText(str(xprev))
                    self.ui.x1it.setText(str(x0))
                    self.loop.exec_()
                print(mp.mpf(x1))

            if (relerror < epsilon):
                xprev = mp.mpf(x0)
                x0 = mp.mpf(x1)
                flag = 1
                self.ui.pushButton_2.hide()

                break
        print('out')
        if (flag == 1):
            self.ui.sol.setText(str(x0))
            self.ui.totalit.setText(str(counter))
            self.ui.pushButton_2.hide()

            return x0
        else:
            self.ui.sol.setText(str(x0))
            self.ui.totalit.setText(str(counter))
            self.ui.message.setText("the method didn't converge for the given iterations")
            print("the method didn't converge for the given iterations")
            self.ui.pushButton_2.hide()

            return x0

    def modifiednewton1(self,StringEquation, iterations, epsilon, initialguess, multplicityOfRoot=1, significantfigures=6):

        flag = 0

        x0 = initialguess

        relerror = 0

        xprev = 0

        counter=0
        if (self.singleStep == True):
            self.ui.x0it.setText(str(x0))
            self.loop.exec_()

        deriv = sympify(diff(StringEquation))

        Equation = sympify(StringEquation)

        mp.dps = significantfigures

        for i in range(1, iterations):

            xprev = mp.mpf(x0)

            counter=i

            fx0 = Equation.evalf(mp.dps, subs={'x': x0})

            fx0_dash = deriv.evalf(mp.dps, subs={'x': x0})

            if (fx0_dash == 0):

                print("Error division by zero\n")
                self.ui.pushButton_2.hide()

                break

            else:

                x1 = mp.mpf(x0 - multplicityOfRoot * (fx0 / fx0_dash))

                relerror = mp.mpf(abs((x1 - xprev) / x1))

                print("relerror", relerror)

                print(f"number={i}"  f"   x0=  {x0}" f"   F(X)=  {fx0}" f"   F(X)`=  {fx0_dash}"  f"     x1=  {x1}")

            if (mp.mpf(abs(Equation.subs("x", x1))) == 0):
                xprev = mp.mpf(x0)
                x0 = mp.mpf(x1)
                flag = 1
                self.ui.pushButton_2.hide()

                break
            else:
                if (self.singleStep == True):
                    self.ui.x0it.setText(str(x0))
                xprev = mp.mpf(x0)
                x0 = mp.mpf(x1)
                if (self.singleStep == True):
                    self.ui.x0it.setText(str(xprev))
                    self.ui.x1it.setText(str(x0))
                    self.loop.exec_()
                print(mp.mpf(x1))

            if (relerror < epsilon):
                flag = 1
                xprev = mp.mpf(x0)
                x0 = mp.mpf(x1)
                self.ui.pushButton_2.hide()

                break
        if (flag == 1):
            self.ui.sol.setText(str(x0))
            self.ui.totalit.setText(str(counter))
            self.ui.pushButton_2.hide()

            return x0
        else:
            self.ui.sol.setText(str(x0))
            self.ui.totalit.setText(str(counter))
            self.ui.message.setText("the method didn't converge for the given iterations")
            print("the method didn't converge for the given iterations")
            self.ui.pushButton_2.hide()

            return x0

    def modifiednewton2(self,StringEquation, iterations, epsilon, initialguess, significantfigures=6):

        flag = 0

        x0 = initialguess

        relerror = 0

        xprev = 0

        counter=0
        if (self.singleStep == True):
            self.ui.x0it.setText(str(x0))
            self.loop.exec_()

        deriv = sympify(diff(StringEquation))
        secondDeriv = sympify(diff(deriv))
        Equation = sympify(StringEquation)

        mp.dps = significantfigures

        for i in range(1, iterations):

            counter=i

            xprev = mp.mpf(x0)
            fx0 = Equation.evalf(mp.dps, subs={'x': x0})
            fx0_dash = deriv.evalf(mp.dps, subs={'x': x0})
            fx0_doubleDash = secondDeriv.evalf(mp.dps, subs={'x': x0})
            check = mp.mpf((fx0_dash ** 2 - fx0 * fx0_doubleDash))

            if (check == 0):

                print("Error division by zero\n")
                self.ui.pushButton_2.hide()

                break

            else:

                x1 = mp.mpf(x0 - ((fx0 * fx0_dash) / (fx0_dash ** 2 - fx0 * fx0_doubleDash)))

                relerror = mp.mpf(abs((x1 - xprev) / x1))

                print("relerror", relerror)

                print(f"number={i}"  f"   x0=  {x0}" f"   F(X)=  {fx0}" f"   F(X)`=  {fx0_dash}"  f"     x1=  {x1}")

            if (mp.mpf(abs(Equation.subs("x", x1))) == 0):
                xprev = mp.mpf(x0)
                x0 = mp.mpf(x1)
                self.ui.pushButton_2.hide()

                flag = 1
                break
            else:
                if (self.singleStep == True):
                    self.ui.x0it.setText(str(x0))
                xprev = mp.mpf(x0)
                x0 = mp.mpf(x1)
                if (self.singleStep == True):
                    self.ui.x0it.setText(str(xprev))
                    self.ui.x1it.setText(str(x0))
                    self.loop.exec_()
                print(mp.mpf(x1))

            if (relerror < epsilon):
                flag = 1
                xprev = mp.mpf(x0)
                x0 = mp.mpf(x1)
                self.ui.pushButton_2.hide()

                break
        if (flag == 1):
            self.ui.sol.setText(str(x0))
            self.ui.totalit.setText(str(counter))
            self.ui.pushButton_2.hide()
            return x0
        else:
            self.ui.pushButton_2.hide()

            self.ui.sol.setText(str(x0))
            self.ui.totalit.setText(str(counter))
            self.ui.message.setText("the method didn't converge for the given iterations")
            print("the method didn't converge for the given iterations")
            return x0

    def falsePosition(self,StringEquation, xl, xu, iterations, epsilon, significantfigures):
        mp.dps = significantfigures
        xprev = mp.mpf(0.0)
        xr = mp.mpf(0.0)
        relerror = mp.mpf(0.0)
        flag = 0
        equation = sympify(StringEquation)
        if (equation.subs('x', xl) * equation.subs('x', xu) > 0):
            self.ui.message.setText("these intervals don't have a root")
            print("these intervals don't have a root")
            return 0
        for i in range(iterations):
            xprev = mp.mpf(xr)
            fxl = equation.evalf(mp.dps, subs={'x': xl})
            fxu = equation.evalf(mp.dps, subs={'x': xu})
            if (fxu - fxl == 0):
                self.ui.message.setText("error division by zero")
                print("error division by zero")
                return 0
            xr = mp.mpf((xl * fxu - xu * fxl) / (fxu - fxl))
            fxr = equation.evalf(mp.dps, subs={'x': xr})
            relerror = mp.mpf(abs((xr - xprev) / xr))
            print("relativeError", relerror)
            print(
                f"number={i}"  f"   xl=  {xl}" f"   xu={xu}" f"   F(Xl)=  {fxl}" f"   F(Xu)`=  {fxu}"  f"     xr=  {xr}")

            if (fxr * fxl == 0):
                self.ui.sol.setText(str(mp.mpf(xr)))
                self.ui.totalit.setText(str(i))
                return mp.mpf(xr)

            elif (fxr * fxl > 0):
                xl = mp.mpf(xr)
            elif (fxr * fxl < 0):
                xu = mp.mpf(xr)
            if (relerror < epsilon):
                flag = 1
                self.ui.sol.setText(str(mp.mpf(xr)))
                self.ui.totalit.setText(str(i))
                return mp.mpf(xr)
        if (flag == 0):
            print("the method didn't converge for the criterea in these iterations ")
            self.ui.message.setText("the method didn't converge for the criterea in these iterations ")
            return mp.mpf(xr)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Solver()
    window.show()
    sys.exit(app.exec_())

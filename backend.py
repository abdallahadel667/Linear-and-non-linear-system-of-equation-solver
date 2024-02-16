import sys

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLineEdit, QLabel
from PyQt5.QtCore import QTimer, QUrl,QEventLoop
from PyQt5.QtMultimedia import QSound, QMediaPlayer, QMediaContent
from PyQt5 import QtGui, QtCore
from frontend import Ui_MainWindow
from iterative import Ui_Iterative
from lu import Ui_luwid
from gausswid import Ui_gausswidget
import copy
import math
from mpmath import mp, matrix
import time

class gausswindow(QWidget):
    def __init__(self,solver):
        super().__init__()
        self.gauss = Ui_gausswidget()
        self.gauss.setupUi(self)
        self.gauss.next.hide()
        self.loop=QEventLoop()
        self.gauss.next.clicked.connect(self.loop.quit)



class luWindow(QWidget):
    def __init__(self,solver):
        super().__init__()
        self.lu = Ui_luwid()
        self.lu.setupUi(self)
        self.lu.next.hide()
        self.loop=QEventLoop()
        self.lu.next.clicked.connect(self.loop.quit)



class IterativeWindow(QWidget):
    def __init__(self,solver):
        super().__init__()
        self.iterative = Ui_Iterative()
        self.iterative.setupUi(self)
        self.ui=solver
        self.loop=QEventLoop()
        self.next=False
        self.iterative.next.hide()
        self.iterative.next.clicked.connect(self.loop.quit)

    # def toggle(self,index):
    #     self.iterative.error.hide()
    #     self.iterative.errorlabel.hide()
    #     self.iterative.iteration.hide()
    #     self.iterative.iterationlabel.hide()
    #     if index ==0:
    #         self.iterative.iteration.show()
    #         self.iterative.iterationlabel.show()
    #     else:
    #         self.iterative.error.show()
    #         self.iterative.errorlabel.show()
    #







class Solver1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Solver")
        self.iterative=IterativeWindow(self)
        self.iterative=IterativeWindow(self)
        self.lu=luWindow(self)
        self.gausswindow=gausswindow(self)
        self.row = int(self.ui.numunknown.currentText())
        self.column = int(self.ui.numunknown.currentText())
        self.augmatrix = [[0] * (self.column+1) for _ in range(self.row)]
        self.matrix=[[0] * (self.column) for _ in range(self.row)]
        self.bmatrix=[0] *(self.row)
        self.solution=[]
        self.initialguess=[0]*(self.row)
        self.num_iteration=self.iterative.iterative.iteration.value()
        self.error=float(self.iterative.iterative.error.text())
        self.singleStep=False
        self.next=False
        self.sig=self.ui.significant.value()
        self.loop = QEventLoop()  # Create an event loop
        self.ui.LUdrop.hide()
        self.ui.conditions.hide()
        self.constants=['X','Y','Z','U','V','W','T','Q']
        self.updateconstants(self.lu.lu.solutionconstgrid)
        self.updateconstants(self.gausswindow.gauss.solutionconstgrid)
        self.updateconstants(self.ui.unknowngrid)
        self.updateconstants(self.ui.solutionconstgrid)
        self.update1d(self.ui.bgrid,0,None)
        self.update1d(self.gausswindow.gauss.solutiongrid,1,None)
        self.update1d(self.lu.lu.solutiongrid,1,None)
        self.update1d(self.ui.solutiongrid,1,None)
        self.update1d(self.iterative.iterative.guessgrid,0,None)
        self.update1d(self.iterative.iterative.iterativesol,1,None)
        self.updateconstants(self.iterative.iterative.iterativeconst)
        self.update2d(self.lu.lu.Lgrid,1,None,True,False)
        self.update2d(self.lu.lu.Ugrid,1,None,True,False)
        self.update2d(self.gausswindow.gauss.auggrid,1,None,True,True)

        self.update2d(self.ui.maingrid,0,None,True,False)

        self.ui.back.clicked.connect(self.tohome)
        self.ui.conditions.clicked.connect(self.showit)
        self.ui.methods.currentIndexChanged.connect(self.methodSelect)
        self.ui.checkBox.stateChanged.connect(self.checkbox)
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.update2d(self.ui.maingrid,0,None,True,False))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.update2d(self.lu.lu.Lgrid,1,None,True,False))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.update2d(self.lu.lu.Ugrid,1,None,True,False))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.ui.bgrid,0,None))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.updateconstants(self.ui.unknowngrid))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.updateconstants(self.ui.solutionconstgrid))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.ui.solutiongrid,1,None))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.updateconstants(self.iterative.iterative.iterativeconst))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.updateconstants(self.lu.lu.solutionconstgrid))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.iterative.iterative.guessgrid,0,None))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.iterative.iterative.iterativesol,1,None))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.lu.lu.solutiongrid,1,None))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.update1d(self.gausswindow.gauss.solutiongrid,1,None))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.updateconstants(self.gausswindow.gauss.solutionconstgrid))
        self.ui.numunknown.currentIndexChanged.connect(lambda :self.update2d(self.gausswindow.gauss.auggrid,1,None,True,True))




        self.iterative.iterative.exit.clicked.connect(self.hideit)
        self.ui.significant.valueChanged.connect(self.figures)
        self.ui.start.clicked.connect(self.start)


    def tohome(self):
        from mainback import Home
        self.home=Home()
        self.home.show()
        self.close()
    def scaled_pivoting(self,matrix, currentRow):
        numRows = self.row
        numCols = self.column+1
        # Find the row with the maximum scaled absolute value in the current column
        maxRowIndex = currentRow
        maxScaledValue = abs(matrix[currentRow, currentRow]) / max(matrix[currentRow, :])

        for k in range(currentRow + 1, numRows):
            scaledValue = abs(matrix[k, currentRow]) / max(matrix[k, :])
            # Set max index to the new max index
            if scaledValue > maxScaledValue:
                maxScaledValue = scaledValue
                maxRowIndex = k
        # Swap the rows to ensure the pivot element has the maximum scaled absolute value
        matrix[currentRow, :], matrix[maxRowIndex, :] = matrix[maxRowIndex, :].copy(), matrix[currentRow, :].copy()
    def pivoting(self,matrix, currentRow):
        numRows = self.row
        # Find the row with the maximum absolute value in the current column
        maxRowIndex = currentRow
        # assume the current pivot is the largest
        maxRowValue = abs(matrix[currentRow,currentRow])
        for k in range(currentRow + 1, numRows):
            currentValue = abs(matrix[k,currentRow])
            # set max index to the new max index
            if currentValue > maxRowValue:
                maxRowValue = currentValue
                maxRowIndex = k
        # Swap the rows to ensure the pivot element has the maximum absolute value
        matrix[currentRow,:], matrix[maxRowIndex,:] = matrix[maxRowIndex,:].copy(), matrix[currentRow,:].copy()

    def gauss(self):
        mp.dps=self.ui.significant.value()
        matrix=copy.deepcopy(self.augmatrix)
        matrix=mp.matrix(matrix)
        numRows = self.row
        numCols = self.column+1
        count_0 = 0
        same=0
        pivot=None
        print('lol')
        for i in range(self.row):
            count_0 = 0
            for j in range(self.column + 1):
                if (matrix[i, j] == 0):
                    count_0 += 1
            if count_0 == self.column + 1:
                print(count_0)
                same = 1
                break
        print(same)
        if same == 0:
            ratio = matrix[0, 0] / matrix[1, 0]
            for i in range(numRows - 1):
                for j in range(numCols):
                    if matrix[0, j] / matrix[1 + i, j] == ratio:
                        same = 1
                    else:
                        same = 0
                        break
                if same != 1:
                    break
        print(same)
        if same ==0:
            if (self.singleStep == True):
                self.update2d(self.gausswindow.gauss.auggrid,1,matrix,False,True)
                self.gausswindow.loop.exec_()
            for i in range(numRows):
                # Partial Pivoting
                if self.ui.scale.isChecked()==True:
                    self.scaled_pivoting(matrix,i)
                else:
                    self.pivoting(matrix, i)
                if (self.singleStep == True):
                    self.update2d(self.gausswindow.gauss.auggrid, 1, matrix, False, True)
                    self.gausswindow.loop.exec_()
                # forward elimination
                pivot = matrix[i,i]
                if pivot==0:
                    break
                for j in range(i + 1, numRows):
                    factor = matrix[j,i] / pivot
                    for k in range(i, numCols):
                        matrix[j,k] -= factor * matrix[i,k]
                    if (self.singleStep == True):
                        self.update2d(self.gausswindow.gauss.auggrid, 1, matrix, False, True)
                        self.gausswindow.loop.exec_()
        # Backward substitution
        print(same,'pivot')
        if pivot!=0 and same==0:
            solution = mp.matrix([0] * numRows)
            for i in range(numRows - 1, -1, -1):
                solution[i] = matrix[i,numCols - 1] / matrix[i,i]
                for j in range(i + 1, numRows):
                    solution[i] -= matrix[i,j] * solution[j] / matrix[i,i]
        self.gausswindow.gauss.next.hide()
        self.gausswindow.closeEvent = None
        count_0 = 0
        countnum = 0
        if same==0:
            for i in range(self.row):
                    count_0 = 0
                    for j in range(self.column + 1):
                        if (matrix[i, j] == 0):
                            count_0 += 1
                    if count_0 == self.column + 1:
                        same = 1
                        break
        if (same == 1):
            for i in range(numRows):
                item = self.ui.solutiongrid.itemAtPosition(i, 0)
                item.widget().setText('Infinite sol.')

        elif (pivot != 0):
            self.solution = copy.deepcopy(solution)
            self.update1d(self.gausswindow.gauss.solutiongrid, 1, self.solution)
            for i in range(numRows):
                item = self.ui.solutiongrid.itemAtPosition(i, 0)
                item.widget().setText(str(self.solution[i]))

        else:
            for i in range(numRows):
                item = self.ui.solutiongrid.itemAtPosition(i, 0)
                item.widget().setText('No solution')

    def getTranspose(self, lowerMatrix):
        mp.dps=self.ui.significant.value()
        n=self.row
        L = [[0] * n for _ in range(n)]
        L=mp.matrix(L)

        for i in range(n):
            for j in range(n):
                L[i,j] = lowerMatrix[j,i]
        return L
    def checkSymmetricity(self):

        for i in range(self.row):
            for j in range(self.row):
                if (self.matrix[i][j] != self.matrix[j][i]):
                    return False
        return True

    def checkPositiveDefinition(self):
        matrix=copy.deepcopy(self.matrix)
        eigen = np.linalg.eig(matrix)
        for eigenValue in eigen.eigenvalues:
            if (eigenValue <= 0):
                return False
        return True
    def sigma(self,i, k, L):
        sum = 0
        for j in range(i):
            sum += (L[i,j] * L[k,j])
        return sum

    def cholesky(self):
        mp.dps=self.ui.significant.value()
        symmetric = self.checkSymmetricity()
        positive_definite = self.checkPositiveDefinition()
        matrix=copy.deepcopy(self.matrix)
        matrix=mp.matrix(matrix)
        if (symmetric and positive_definite):
            L = [[0] * self.row for _ in range(self.row)]
            L=mp.matrix(L)

            if self.singleStep == True:

                self.lu.closeEvent = lambda event: event.ignore()
                self.lu.lu.next.show()
                self.lu.show()

            for k in range(self.row):
                if self.singleStep==True and k==0:
                    self.update2d(self.lu.lu.Lgrid, 1, L, False,False)
                    self.update2d(self.lu.lu.Ugrid, 1, None, False,False)
                    self.lu.loop.exec_()
                for i in range(k + 1):
                    Sigma = self.sigma(i, k, L)
                    if (k == i):
                        L[k,k] = math.sqrt(matrix[k,k] - Sigma)
                    else:
                        L[k,i] = (matrix[k,i] - Sigma) / L[i,i]
                    if self.singleStep==True:
                        self.update2d(self.lu.lu.Lgrid, 1, L,False,False)
                        self.lu.loop.exec_()


            LT = self.getTranspose(L)
            if self.singleStep==True:
               self.update2d(self.lu.lu.Ugrid, 1, LT,False,False)
               self.lu.loop.exec_()

            B=copy.deepcopy(self.bmatrix)
            B=mp.matrix(B)
            Z = [0 for j in range(self.row)]
            Z=mp.matrix(Z)
            for i in range(0, self.row):
                for j in range(1, i + 1):
                    B[i] = B[i] - (L[i,j - 1] * Z[j - 1])
                Z[i] = B[i] / L[i,i]
                # print(Z[i])
            # To get the X matrix
            X = [0 for j in range(self.row)]
            X=mp.matrix(X)
            for i in range(self.row - 1, -1, -1):
                sum_term = 0
                for j in range(i + 1, self.row):
                    sum_term = sum_term + (LT[i,j] * X[j])
                X[i] = (Z[i] - sum_term) / LT[i,i]



            if self.singleStep == True:
                 self.update1d(self.lu.lu.solutiongrid, 1, X)
            self.solution = copy.deepcopy(X)
            self.lu.lu.next.hide()
            self.lu.closeEvent = None
            for i in range(self.row):
                item = self.ui.solutiongrid.itemAtPosition(i, 0)
                item.widget().setText(str(self.solution[i]))

        elif (symmetric):
            print("Cholesky decomposition is not allowed for non-positive definite matrix")
        else:
            print("Cholesky decomposition is not allowed for unsymmetric")


    def crout(self):
        mp.dps=self.ui.significant.value()
        matrix=copy.deepcopy(self.matrix)
        matrix=mp.matrix(matrix)
        carryU=copy.deepcopy(self.matrix)
        carryU=mp.matrix(carryU)
        n=self.row
        # matrix = [[0 for j in range(n)] for i in range(n)]
        # carryU = [[0 for j in range(n)] for i in range(n)]
        L = [[0 for j in range(n)] for i in range(n)]
        U = [[0 for j in range(n)] for i in range(n)]
        L=mp.matrix(L)
        U=mp.matrix(U)
        B=copy.deepcopy(self.bmatrix)
        B=mp.matrix(B)

        for i in range(n):
            B[i] = float(B[i])
            for j in range(n):
                matrix[i,j] = float(matrix[i,j])
                carryU[i,j] = float(carryU[i,j])

        for k in range(n):
            for i in range(k, n):
                sum_term = 0
                for j in range(k):
                    sum_term = sum_term + (L[i,j] * U[j,k])
                L[i,k] = carryU[i,k] - sum_term
                if self.singleStep==True:
                    self.update2d(self.lu.lu.Lgrid,1,L,False,False)
                    self.lu.loop.exec_()

            for i in range(k + 1, n):
                sum_term = 0
                for j in range(k):
                    sum_term = sum_term + (L[k,j] * U[j,i])
                U[k,i] = (carryU[k,i] - sum_term) / L[k,k]
                if self.singleStep==True:
                    self.update2d(self.lu.lu.Ugrid,1,U,False,False)
                    self.lu.loop.exec_()

        # create the U matrix and then print it
        for i in range(0, n):
            for j in range(0, n):
                #  U[i][j]=carryU[i][j]
                if i == j:
                    U[i,j] = 1
        if self.singleStep == True:
            self.update2d(self.lu.lu.Ugrid, 1, U,False,False)
            self.lu.loop.exec_()


        Z = [0 for j in range(n)]
        Z=mp.matrix(Z)
        for i in range(0, n):
            for j in range(1, i + 1):
                B[i] = B[i] - (L[i,j - 1] * Z[j - 1])
            Z[i] = B[i] / L[i,i]
            # print(Z[i])

        # To get the X matrix
        X = [0 for j in range(n)]
        X=mp.matrix(X)
        for i in range(n):
            X[i] = float(X[i])
        for i in range(n - 1, -1, -1):
            sum_term = 0
            for j in range(i + 1, n):
                sum_term = sum_term + (U[i,j] * X[j])
            X[i] = (Z[i] - sum_term)
        print(X)
        if self.singleStep==True:
            self.update1d(self.lu.lu.solutiongrid,1,X)

        self.solution = copy.deepcopy(X)
        self.lu.lu.next.hide()
        self.lu.closeEvent=None
        for i in range(self.row):
            item = self.ui.solutiongrid.itemAtPosition(i, 0)
            item.widget().setText(str(self.solution[i]))

    def doolittle(self):
        mp.dps=self.ui.significant.value()
        matrix=copy.deepcopy(self.matrix)
        matrix=mp.matrix(matrix)
        carryU=copy.deepcopy(self.matrix)
        carryU=mp.matrix(carryU)

        B=copy.deepcopy(self.bmatrix)
        B=mp.matrix(B)
        print(self.bmatrix)
        n=self.row
        L = [[0 for j in range(n)] for i in range(n)]
        U = [[0 for j in range(n)] for i in range(n)]
        L=mp.matrix(L)
        U=mp.matrix(U)
        for i in range(n):
            #self.pivoting(matrix,i)
            B[i] = float(B[i])
            for j in range(n):
                matrix[i,j] = float(matrix[i,j])
                carryU[i,j] = float(carryU[i,j])
        if self.singleStep == True:
            self.update2d(self.lu.lu.Lgrid, 1, L,False,False)
            self.lu.loop.exec_()
        for k in range(n):
            L[k,k] = 1  # Set diagonal elements of L to 1
            for i in range(k + 1, n):
                factor = carryU[i,k] / carryU[k,k]
                L[i,k] = factor
                if self.singleStep==True:
                    self.update2d(self.lu.lu.Lgrid,1,L,False,False)
                    self.lu.loop.exec_()
                for j in range(k, n):
                    carryU[i,j] -= factor * carryU[k,j]
        if self.singleStep == True:
            self.update2d(self.lu.lu.Lgrid, 1, L,False,False)
            self.lu.loop.exec_()
        # create the U matrix and then print it
        for i in range(0, n):
            for j in range(0, n):
                U[i,j] = carryU[i,j]
                if self.singleStep==True:
                    self.update2d(self.lu.lu.Ugrid,1,U,False,False)
                    self.lu.loop.exec_()

        Z = [0 for j in range(n)]
        Z=mp.matrix(Z)
        for i in range(0, n):
            for j in range(1, i + 1):
                B[i] = B[i] - (L[i,j - 1] * Z[j - 1])
            Z[i] = B[i] / L[i,i]
            # print(Z[i])

        X = [0 for j in range(n)]
        X=mp.matrix(X)
        for i in range(n - 1, -1, -1):
            sum_term = 0
            for j in range(i + 1, n):
                sum_term = sum_term + (U[i,j] * X[j])
            X[i] = (Z[i] - sum_term) / U[i,i]
        if self.singleStep==True:
            self.update1d(self.lu.lu.solutiongrid,1,X)
        self.solution = copy.deepcopy(X)
        self.lu.closeEvent=None
        self.lu.lu.next.hide()
        for i in range(self.row):
            item = self.ui.solutiongrid.itemAtPosition(i, 0)
            item.widget().setText(str(self.solution[i]))

    def is_diagonally_dominant(self):
            matrix=copy.deepcopy(self.matrix)
            print(matrix)

            for i in range(self.row):
                print('here')
                diagonal_element = abs(matrix[i][i])
                row_sum = np.sum(np.abs(matrix[i])) - diagonal_element

                if diagonal_element <= row_sum:
                    print('here32')
                    return False

            return True
    def seidel(self):

        mp.dps=self.ui.significant.value()
        arr = copy.deepcopy(self.matrix)
        dominant = self.is_diagonally_dominant()
        arr=mp.matrix(arr)
        numit = self.num_iteration
        absolrelerror = self.error
        coeffArr = copy.deepcopy(self.initialguess)
        coeffArr=mp.matrix(coeffArr)
        equals = copy.deepcopy(self.bmatrix)
        equals=mp.matrix(equals)
        newcoeff = copy.deepcopy(coeffArr)
        newcoeff=mp.matrix(newcoeff)
        rowlen = self.row
        err = [10000000000] * rowlen
        err = mp.matrix(err)
        flag = True
        l = 0
        if self.singleStep == True:
            self.update1d(self.iterative.iterative.iterativesol, 1, coeffArr)
            self.iterative.loop.exec_()
        for p in range(numit):
                for i in range(0, rowlen):
                    aproxval = equals[i]
                    real = equals[i]  # making it equal the first value in the num arra
                    for j in range(0, rowlen):
                        if (i != j):
                            aproxval -= arr[i,j] * newcoeff[j]  # making our calculations from the equations given and subtracting them from our real value for the approximate
                    newcoeff[i] = aproxval / arr[i,i]  # dividing on the coeffectient of the calculated variable of the equation
                for m in range(0, rowlen):
                    if(newcoeff[m]==0):
                        continue
                    err[m] = abs(
                        (newcoeff[m] - coeffArr[m]) / newcoeff[m]) * 100  # calculating the absolute relative error
                print(err)
                for n in range(0, rowlen):
                    if (err[n] < absolrelerror):
                        flag = False
                    else:
                        flag = True
                        break
                coeffArr = copy.deepcopy(newcoeff)
                if self.singleStep==True:
                    self.update1d(self.iterative.iterative.iterativesol,1,coeffArr)
                    self.iterative.loop.exec_()
                # updating the coeffecients
                if flag == False:
                    break
        if dominant==False:
            self.iterative.iterative.converge.setText("The values didn't converge as the matrix isn't diagonally dominant")
            self.ui.label_15.setText("The values didn't converge as the matrix isn't diagonally dominant")


        elif p==numit-1:
          self.iterative.iterative.converge.setText("The values didn't converge for the given number of iterations")
          self.ui.label_15.setText("The values didn't converge for the given number of iterations")

        # if absolrelerror is None:
        #     if self.singleStep == True:
        #         self.update1d(self.iterative.iterative.iterativesol, 1, coeffArr)
        #         self.iterative.loop.exec_()
        #     for s in range(0, numit):
        #         for i in range(0, rowlen):
        #             aproxval = equals[i]  # making it equal the first value in the num array for the approximation
        #             for j in range(0, rowlen):
        #                 if (i != j):
        #                     aproxval -= round(arr[i][j] * coeffArr[j], 3)
        #             newcoeff[i] = aproxval / arr[i][i]
        #             coeffArr = newcoeff.copy()
        #         if self.singleStep == True:
        #             self.update1d(self.iterative.iterative.iterativesol, 1, coeffArr)
        #             self.iterative.loop.exec_()
        self.solution = copy.deepcopy(coeffArr)
        self.iterative.iterative.next.hide()
        self.iterative.closeEvent = None
        for i in range(self.row):
            item = self.ui.solutiongrid.itemAtPosition(i, 0)
            item.widget().setText(str(self.solution[i]))
    def jacobi(self):
        mp.dps=self.ui.significant.value()
        arr=copy.deepcopy(self.matrix)
        dominant=self.is_diagonally_dominant()

        arr=mp.matrix(arr)
        numit=self.num_iteration
        absolrelerror=self.error
        coeffArr=copy.deepcopy(self.initialguess)
        coeffArr=mp.matrix(coeffArr)
        equals=copy.deepcopy(self.bmatrix)
        equals=mp.matrix(equals)
        newcoeff = copy.deepcopy(coeffArr)
        newcoeff=mp.matrix(newcoeff)
        rowlen = self.row
        err = [10000000000] * rowlen
        err=mp.matrix(err)
        flag = True
        l = 0
        x = [0] * rowlen
        x=mp.matrix(x)
        if self.singleStep == True:
            self.iterative.closeEvent = lambda event: event.ignore()
            self.update1d(self.iterative.iterative.iterativesol, 1, coeffArr)
            self.iterative.loop.exec_()
        for p in range(numit):
                for i in range(0, rowlen):
                    aproxval = equals.copy()
                    for j in range(0, rowlen):
                        if (i != j):
                            aproxval[i] -= (arr[i,j] * coeffArr[j])
                    x[i] = aproxval[i]
                for n in range(0, rowlen):
                    newcoeff[n] = x[n] / arr[n,n]
                for m in range(0, rowlen):
                    if(newcoeff[m]==0):
                        continue
                    err[m] = abs((newcoeff[m] - coeffArr[m]) / newcoeff[m]) * 100
                for m in range(0, rowlen):
                    if (err[m] < absolrelerror):
                        flag = False
                    else:
                        flag = True
                        break
                coeffArr = copy.deepcopy(newcoeff)
                print(coeffArr)
                if self.singleStep==True:
                    self.update1d(self.iterative.iterative.iterativesol,1,coeffArr)
                    self.iterative.loop.exec_()
                if flag == False:
                    break

        if dominant == False:
            self.iterative.iterative.converge.setText(
                "The values didn't converge as the matrix isn't diagonally dominant")
            self.ui.label_15.setText("The values didn't converge as the matrix isn't diagonally dominant")


        elif p == numit - 1:
            self.iterative.iterative.converge.setText("The values didn't converge for the given number of iterations")
            self.ui.label_15.setText("The values didn't converge for the given number of iterations")
        self.solution = copy.deepcopy(coeffArr)
        self.iterative.iterative.next.hide()
        self.iterative.closeEvent = None
        for i in range(self.row):
            item = self.ui.solutiongrid.itemAtPosition(i, 0)
            item.widget().setText(str(self.solution[i]))
    def gauss_jordan(self):
        mp.dps=self.ui.significant.value()
        numRows = len(self.augmatrix)
        numCols = len(self.augmatrix[0])
        matrix = copy.deepcopy(self.augmatrix)
        matrix2 = copy.deepcopy(self.augmatrix)
        matrix=mp.matrix(matrix)
        count_0 = 0
        same=0
        for i in range(self.row):
            count_0 = 0
            for j in range(self.column + 1):
                if (matrix2[i][ j] == 0):
                    count_0 += 1
            if count_0 == self.column + 1:
                same = 1
                break
        if same==0:
            ratio=matrix2[0][0]/matrix2[1][0]
            for i in range (numRows-1):
             for j in range (numCols):
                if matrix2[0][j]/matrix2[1+i][j]==ratio:
                    same=1
                else:
                    same=0
                    break
             if same!=1:
                 break
        if same==0:
            if (self.singleStep == True):
                self.update2d(self.gausswindow.gauss.auggrid,1,matrix,False,True)
                self.gausswindow.loop.exec_()
            for i in range(numRows):

                if self.ui.scale.isChecked()==True:
                    self.scaled_pivoting(matrix,i)
                else:
                    self.pivoting(matrix,i)
                if(self.singleStep==True):
                    self.update2d(self.gausswindow.gauss.auggrid, 1, matrix, False, True)
                    self.gausswindow.loop.exec_()
                pivot = matrix[i,i]
                if pivot==0:
                    break
                for j in range(i + 1, numRows):
                    factor = matrix[j,i] / pivot
                    for k in range(i, numCols):
                        matrix[j,k] -= factor * matrix[i,k]
                    if (self.singleStep == True):
                        self.update2d(self.gausswindow.gauss.auggrid, 1, matrix, False, True)
                        self.gausswindow.loop.exec_()

                        # Backward elimination
            print(pivot,same)
            if pivot!=0 and same==0:
                for i in range(numRows - 1, -1, -1):
                    pivot = matrix[i,i]

                    # Normalizing values
                    for k in range(numCols - 1, i - 1, -1):
                        matrix[i,k] /= pivot
                    if (self.singleStep == True):
                        self.update2d(self.gausswindow.gauss.auggrid, 1, matrix, False, True)
                        self.gausswindow.loop.exec_()

                    for j in range(i - 1, -1, -1):
                        factor = matrix[j,i]
                        for k in range(i, numCols):
                            matrix[j,k] -= factor * matrix[i,k]
                        if (self.singleStep == True):
                            self.update2d(self.gausswindow.gauss.auggrid, 1, matrix, False, True)
                            self.gausswindow.loop.exec_()

                            # Extract solutions
        self.gausswindow.gauss.next.hide()
        self.gausswindow.closeEvent = None
        if same==0:
            for i in range(self.row):
                count_0 = 0
                for j in range(self.column + 1):
                    if (matrix[i, j] == 0):
                        count_0 += 1
                if count_0 == self.column + 1:
                    print(count_0)
                    same = 1
                    break
        if(same==1):
            for i in range(numRows):
                item = self.ui.solutiongrid.itemAtPosition(i, 0)
                item.widget().setText('Infinite sol.')

        elif(pivot!=0):
            solution = mp.matrix([0] * numRows)
            solution = mp.matrix([[matrix[row, numCols - 1]] for row in range(numRows)])
            print(solution)

            self.solution = copy.deepcopy(solution)
            print(self.solution)
            self.update1d(self.gausswindow.gauss.solutiongrid,1,self.solution)
            for i in range(numRows):
                item = self.ui.solutiongrid.itemAtPosition(i, 0)
                item.widget().setText(str(self.solution[i]))

        else:
            for i in range(numRows):
                item = self.ui.solutiongrid.itemAtPosition(i, 0)
                item.widget().setText('No solution')
    def start(self):
        self.updateMatrix()
        self.iterative.iterative.converge.setText("")
        self.ui.label_15.setText("")
        if self.ui.methods.currentIndex()==0:
            if self.singleStep == True:
                self.gausswindow.closeEvent = lambda event: event.ignore()
                self.gausswindow.gauss.next.show()
                self.gausswindow.show()
            start = time.time()
            self.gauss()
        if self.ui.methods.currentIndex()==1:

            if self.singleStep == True:
                self.gausswindow.closeEvent = lambda event: event.ignore()
                self.gausswindow.gauss.next.show()
                self.gausswindow.show()
            start = time.time()
            self.gauss_jordan()
        if self.ui.methods.currentIndex()==2 and self.ui.LUdrop.currentIndex()==0:
            if self.singleStep == True:
                self.lu.closeEvent = lambda event: event.ignore()
                self.lu.lu.next.show()
                self.lu.show()
            start = time.time()
            self.doolittle()
        if self.ui.methods.currentIndex()==2 and self.ui.LUdrop.currentIndex()==1:
            if self.singleStep == True:
                self.lu.closeEvent = lambda event: event.ignore()
                self.lu.lu.next.show()
                self.lu.show()
            start = time.time()
            self.crout()
        if self.ui.methods.currentIndex() == 2 and self.ui.LUdrop.currentIndex() == 2:
            start = time.time()
            self.cholesky()
        if self.ui.methods.currentIndex()==3:
            if self.singleStep == True:
                self.iterative.closeEvent = lambda event: event.ignore()
                self.iterative.iterative.next.show()
                self.iterative.show()
                self.iterative.iterative.exit.hide()
            self.update_iterative()
            start = time.time()
            self.seidel()
        if self.ui.methods.currentIndex()==4:
            if self.singleStep == True:
                print('ignore')
                self.iterative.closeEvent = lambda event: event.ignore()
                self.iterative.iterative.next.show()
                self.iterative.show()
                self.iterative.iterative.exit.hide()

            self.update_iterative()
            start = time.time()
            self.jacobi()
        end=time.time()
        print((end-start))

        self.ui.time.setText(str(round(end-start,10)))
    def methodSelect(self,index):
        self.ui.LUdrop.hide()
        self.ui.conditions.hide()
        self.ui.scale.hide()
        if index==0 or index==1:
            self.ui.scale.show()
        if index==2:
            self.ui.LUdrop.show()
        elif index==3 or index==4:
            self.ui.conditions.show()
    def update_iterative(self):
        self.num_iteration = self.iterative.iterative.iteration.value()
        self.error = float(self.iterative.iterative.error.text())
        print(self.num_iteration)
        print(self.error)
        print(self.row)
        for i in range(self.row):
            item=self.iterative.iterative.guessgrid.itemAtPosition(i,0)
            text = item.widget().text()
            print('osama')
            # self.initialguess[i]= int(text) if text.lstrip('-').isdigit() else 0
            try:
                float_value = float(text)
                print('yoyssef')
                self.initialguess[i]= float_value

            except ValueError:
                self.initialguess[i]=0
        print('out')
    def updateconstants(self,grid):
        self.cleargrid(grid)
        self.row = int(self.ui.numunknown.currentText())
        self.column = int(self.ui.numunknown.currentText())
        for j in range(self.column):
            label = QLabel(f'{self.constants[j]}')
            font = QtGui.QFont()
            font.setPointSize(16-self.column)
            font.setBold(True)
            label.setFont(font)
            label.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(label,j, 0)

    def update1d(self,grid,value,matrix):
        self.cleargrid(grid)
        self.row = int(self.ui.numunknown.currentText())
        self.column = int(self.ui.numunknown.currentText())
        for j in range(self.column):
            if matrix is None:
              line_edit = QLineEdit(f'0')
            else:
              line_edit=QLineEdit(f'{matrix[j]}')
            font = QtGui.QFont()
            font.setPointSize(14-self.column)
            line_edit.setFont(font)
            line_edit.setAlignment(QtCore.Qt.AlignCenter)
            if value==1:
                line_edit.setReadOnly(True)
                line_edit.setFrame(False)
            grid.addWidget(line_edit,j, 0)

    def update2d(self,grid,value,matrix,clear,aug):
        self.cleargrid(grid)
        self.row = int(self.ui.numunknown.currentText())
        self.column = int(self.ui.numunknown.currentText())
        column=self.column
        if aug==True:
            column=column+1

        if clear==True:
            self.augmatrix = [[0] * (self.column + 1) for _ in range(self.row)]
            self.matrix=[[0] * (self.column) for _ in range(self.row)]
            self.bmatrix=[0] *(self.row)
            self.initialguess = [0] * (self.row)

        for i in range(self.row):
            for j in range(column):
                if matrix is not None:
                    try:
                      line_edit=QLineEdit(f'{matrix[i][j]}')
                    except (IndexError, TypeError):
                      line_edit = QLineEdit(f'{matrix[i,j]}')
                else:
                    line_edit = QLineEdit(f'0')
                font = QtGui.QFont()
                font.setPointSize(14-self.column)
                line_edit.setFont(font)
                line_edit.setAlignment(QtCore.Qt.AlignCenter)
                if value == 1:
                    line_edit.setReadOnly(True)
                    line_edit.setFrame(False)
                grid.addWidget(line_edit, i, j)


    def update_preview(self,matrix):
        self.ui.lineEdit_7.setText(str(matrix[0][0]))
        self.ui.lineEdit_8.setText(str(matrix[0][1]))
        self.ui.lineEdit_9.setText(str(matrix[1][0]))
        self.ui.lineEdit_10.setText(str(matrix[1][1]))
        self.ui.lineEdit_11.setText(str(matrix[0][2]))
        self.ui.lineEdit_12.setText(str(matrix[1][2]))

    def togglenext(self):
        self.next=True

    def checkbox(self):
        self.singleStep=not self.singleStep

    def cleargrid(self,grid):
        while grid.count():
          item = grid.takeAt(0)
          widget = item.widget()
          if widget:
            widget.deleteLater()
    # def updateMatrix(self):
    #     for i in range(self.row):
    #         self.bmatrix[i] = float(self.bmatrix[i])
    #         for j in range(self.row):
    #             self.matrix[i][j] = float(self.matrix[i][j])
    #     for i in range(self.row):
    #         for j in range(self.row+1):
    #             self.augmatrix[i][j] = float(self.augmatrix[i][j])
    #     for i in range(self.row):
    #         for j in range(self.column):
    #             item=self.ui.maingrid.itemAtPosition(i,j)
    #             text=item.widget().text()
    #             self.augmatrix[i][j] = float(text) if text.lstrip('-').isdigit() else 0
    #             self.matrix[i][j] = float(text) if text.lstrip('-').isdigit() else 0
    #
    #     for i in range(self.row):
    #         item=self.ui.bgrid.itemAtPosition(i,0)
    #         text = item.widget().text()
    #         self.augmatrix[i][self.column] = float(text) if text.lstrip('-').isdigit() else 0
    #         self.bmatrix[i] = float(text) if text.lstrip('-').isdigit() else 0
    #     print(self.bmatrix)
    #     print(self.matrix)
    #     print(self.augmatrix)
    def updateMatrix(self):
        for i in range(self.row):
            self.bmatrix[i] = float(self.bmatrix[i])
            for j in range(self.row):
                self.matrix[i][j] = float(self.matrix[i][j])
        for i in range(self.row):
            for j in range(self.row + 1):
                self.augmatrix[i][j] = float(self.augmatrix[i][j])
        for i in range(self.row):
            for j in range(self.column):
                item = self.ui.maingrid.itemAtPosition(i, j)
                text = item.widget().text()
                try:
                    float_value = float(text)
                    self.augmatrix[i][j] = float_value
                    self.matrix[i][j] = float_value
                except ValueError:
                    self.augmatrix[i][j] = 0
                    self.matrix[i][j] = 0

        for i in range(self.row):
            item = self.ui.bgrid.itemAtPosition(i, 0)
            text = item.widget().text()
            try:
                float_value = float(text)
                self.augmatrix[i][self.column] = float_value
                self.bmatrix[i] = float_value
            except ValueError:
                self.augmatrix[i][self.column] = 0
                self.bmatrix[i] = 0
    def figures(self):
        self.sig=self.ui.significant.value()
        print(self.sig)
    def showit(self):
        self.iterative.show()
        self.iterative.closeEvent=None
        self.iterative.closeEvent = lambda event: event.ignore()
    def hideit(self):
        self.iterative.hide()






if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Solver1()
    window.show()
    sys.exit(app.exec_())

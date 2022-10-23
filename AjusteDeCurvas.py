import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def AjusteLineal(X,Y,Tabla='N',Grafica='N',Labelx='Valores x',Labely='Valores y'):
    X = np.reshape(X, (-1, 1))  # Si el vector X es fila conviértalo en vector columna
    Y = np.reshape(Y, (-1, 1))  # Si el vector Y es fila conviértalo en vector columna
    # Cree la matriz A
    A = np.append(np.ones(np.shape(X)), X, 1)
    # Ecuaciones normales  A^TA c = A^T y
    ATA = np.matmul(np.transpose(A), A)
    ATy = np.matmul(np.transpose(A), Y)
    # Encuentre los coeficientes c usando np.linalg.solve
    Coefs = np.linalg.solve(ATA, ATy)
    # Encuentre los valores aproximados, los residuales y el error de aproximación.
    Yaprox = np.matmul(A, Coefs)
    Residual= Yaprox-Y
    Error = np.linalg.norm(Residual)
    print('La recta de mejor ajuste es:  y=', Coefs[1].item(), 'x  + ', 
Coefs[0].item())
    print('Error de Mínimos cuadrados', Error)
    print(' ')
    if Tabla == 'S' or Tabla=='Y' or Tabla==1 or Tabla=='Si' or Tabla=='Yes':
        print(' ')
        print('Resultados detallados del Ajuste Lineal')
        print('   x       y      yapprox     residual')
        for i in range(0, len(Y)):
            #print(
            #    '{:3d}  {:3d}    {:6.4f}    {:7.5f}'.format(X[i].item(), Y[i].item(), Yaprox[i].item(), Residual[i].item()))
            print(
                '{:6.1f}  {:6.3f}    {:6.3f}     {:7.5f}'.format(X[i].item(), Y[i].item(), Yaprox[i].item(),
                                                            Residual[i].item()))
    if Grafica=='S' or Grafica=='Y' or Grafica==1 or Grafica=='Si' or Grafica=='Yes':
        #Grafique los datos actuales y los datos aproximados
        plt.plot(X, Y, 'o')
        plt.plot(X, Yaprox, linestyle='dashed')
        plt.xlabel(Labelx)  # rótulo para el eje x
        plt.ylabel(Labely)  # rótulo para el eje y
        plt.legend(['Actuales','Aproximados'])  # Leyenda datos actuales y aproxs
        plt.grid()  # agrega una cuadrícula
        plt.title('Ajuste Lineal y= {:2.3f}x + {:2.3f}'.format(Coefs[1].item(), 
Coefs[0].item()))  # título de la gráfica
        plt.show()
    return Coefs,Yaprox  #Devuelve los coeficientes y los valores aproximados
    #############################
#############################
def AjusteCuadratico(X,Y,Tabla='N',Grafica='N',Labelx='Valores x',Labely='Valores y'):
    # Cree la matriz A
    X=np.reshape(X,(-1,1)) #Si el vector X es fila conviértalo en vector columna
    Y=np.reshape(Y,(-1,1))  # Si el vector Y es fila conviértalo en vector columna
    A = np.append(np.ones(np.shape(X)), X, 1) #Agregue la columna de X's
    A = np.append(A, X**2, 1)  # Agregue la columna de X^2's
    # Ecuaciones normales  A^TA c = A^T y
    ATA = np.matmul(np.transpose(A), A)
    ATy = np.matmul(np.transpose(A), Y)
    # Encuentre los coeficientes c usando np.linalg.solve
    Coefs = np.linalg.solve(ATA, ATy)
    # Encuentre los valores aproximados, los residuales y el error de aproximación.
    Yaprox = np.matmul(A, Coefs)
    Residual= Yaprox-Y
    Error = np.linalg.norm(Residual)
    print('La parábola de mejor ajuste es y= {:2.3f}x^2 + {:2.3f}x + {:2.3f}'. 
format(Coefs[2].item(),Coefs[1].item(), Coefs[0].item()))
    print('Error de Mínimos cuadrados', Error)
    print(' ')
    if Tabla == 'S' or Tabla=='Y' or Tabla==1 or Tabla=='Si' or Tabla=='Yes':
        print(' ')
        print('Resultados detallados del Ajuste Cuadrático')
        print('   x       y      yapprox     residual')
        for i in range(0, len(Y)):
            #print(
            #    '{:3d}  {:3d}    {:6.4f}    {:7.5f}'.format(X[i].item(),Y[i].item(), Yaprox[i].item(), Residual[i].item()))
            print(
                '{:6.1f}  {:6.3f}    {:6.3f}     {:7.5f}'.format(X[i].item(), 
Y[i].item(), Yaprox[i].item(),
                                                            Residual[i].item()))
    if Grafica=='S' or Grafica=='Y' or Grafica==1 or Grafica=='Si' or Grafica=='Yes':
        #Grafique los datos actuales y los datos aproximados
        plt.plot(X, Y, 'o')
        plt.plot(X, Yaprox, linestyle='dashed')
        plt.xlabel(Labelx)  # rótulo para el eje x
        plt.ylabel(Labely)  # rótulo para el eje y
        plt.legend(['Actuales','Aproximados'])  # Leyenda datos actuales y aproxs
        plt.grid()  # agrega una cuadrícula
        plt.title('Ajuste Cuadrático y= {:2.3f}x^2 + {:2.3f}x + {:2.3f}'. 
format(Coefs[2].item(),Coefs[1].item(), Coefs[0].item()) ) #título de la gráfica
        plt.show()
    return Coefs, Yaprox  # Devuelve los coeficientes y los valores aproximados
#############################
#############################
def AjustePolinomial(X,Y,n=1,Tabla='N',Grafica='N',Labelx='Valores x',Labely='Valores y'):
        X = np.reshape(X, (-1, 1))  # Si el vector X es fila conviértalo en vector columna
        Y = np.reshape(Y, (-1, 1))  # Si el vector Y es fila conviértalo en vector columna
        # Cree la matriz A
        A=np.ones(np.shape(X)) #La matriz inicial sólo tiene unos
        for i in range(1,n+1):
            A = np.append(A, X**i, 1) #Agregue la columna de X's
        # Ecuaciones normales  A^TA c = A^T y
        ATA = np.matmul(np.transpose(A), A)
        ATy = np.matmul(np.transpose(A), Y)
        # Encuentre los coeficientes c usando np.linalg.solve
        Coefs = np.linalg.solve(ATA, ATy)
        # Encuentre los valores aproximados, los residuales y el error de aproximación.
        Yaprox = np.matmul(A, Coefs)
        Residual= Yaprox-Y
        Error = np.linalg.norm(Residual)
        print('Polinomio de grado', n, 'de mejor ajuste')
        print('Error de Mínimos cuadrados', Error)
        print(' ')
        if Tabla == 'S' or Tabla=='Y' or Tabla==1 or Tabla=='Si' or Tabla=='Yes':
            print(' ')
            print('Resultados detallados del Ajuste Polinomial grado',n)
            print('   x       y      yapprox     residual')
            for i in range(0, len(Y)):
                #print(
                #    '{:3d}  {:3d}    {:6.4f}    {:7.5f}'.format(X[i].item(), Y[i].item(), Yaprox[i].item(), Residual[i].item()))
                print(
                    '{:6.1f}  {:6.3f}    {:6.3f}     {:7.5f}'.format(X[i].item(), 
    Y[i].item(), Yaprox[i].item(),
                                                                Residual[i].item()))
        if Grafica=='S' or Grafica=='Y' or Grafica==1 or Grafica=='Si' or Grafica=='Yes':
            #Grafique los datos actuales y los datos aproximados
            plt.plot(X, Y, 'o')
            plt.plot(X, Yaprox, linestyle='dashed')
            plt.xlabel(Labelx)  # rótulo para el eje x
            plt.ylabel(Labely)  # rótulo para el eje y
            plt.legend(['Actuales','Aproximados'])  # Leyenda datos actuales y aproxs
            plt.grid()  # agrega una cuadrícula
            plt.title('Ajuste Polinomial de Grado {:2d}'.format(n) ) #título de la gráfica
            plt.show()
        return Coefs, Yaprox  # Devuelve los coeficientes y los valores aproximados
    ###################################
    ###################################
def AjusteExponencial(X,Y,Tabla='N',Grafica='N',Labelx='Valores x',Labely='Valores y'):
    X = np.reshape(X, (-1, 1))  # Si el vector X es fila conviértalo en vector columna
    Y = np.reshape(Y, (-1, 1))  # Si el vector Y es fila conviértalo en vector columna
    # Cree la matriz A
    A = np.append(np.ones(np.shape(X)), X, 1)
    # Ecuaciones normales  A^TA c = A^T y
    ATA = np.matmul(np.transpose(A), A)
    ATlnY = np.matmul(np.transpose(A), np.log(Y))
    # Encuentre los coeficientes c usando np.linalg.solve
    Coefs = np.linalg.solve(ATA, ATlnY)
    # Encuentre los valores aproximados, los residuales y el error de aproximación.
    lnYaprox = np.matmul(A, Coefs)
    Yaprox  =np.exp(lnYaprox)  #Use la función exponencial para encontrar los valores Y aproximados
    Residual= lnYaprox-np.log(Y)
    Error = np.linalg.norm(Residual)
    print('Modelo Exponencial de mejor ajuste:  y=', np.exp(Coefs[0].item()), 'exp(', Coefs[1].item(),'t )')
    print('Error de Mínimos cuadrados', Error)
    print(' ')
    if Tabla == 'S' or Tabla=='Y' or Tabla==1 or Tabla=='Si' or Tabla=='Yes':
        print(' ')
        print('Resultados detallados del Ajuste Exponencial')
        print('   x       ln(y)      ln(yapprox)     residual')
        for i in range(0, len(Y)):
            print(
                '{:6.1f}  {:6.3f}    {:6.3f}     {:7.5f}'.format(X[i].item(), np.log(Y[i].item()), lnYaprox[i].item(),Residual[i].item()))
        print(' ')
    if Grafica=='S' or Grafica=='Y' or Grafica==1 or Grafica=='Si' or Grafica=='Yes':
        #Grafique los datos actuales y los datos aproximados
        plt.plot(X, Y, 'o')
        plt.plot(X, Yaprox, linestyle='dashed')
        plt.xlabel(Labelx)  # rótulo para el eje x
        plt.ylabel(Labely)  # rótulo para el eje y
        plt.legend(['Actuales','Aproximados'])  # Leyenda datos actuales y aproxs
        plt.grid()  # agrega una cuadrícula
        plt.title('Modelo Exponencial y= {:2.3e}exp({:2.3e}t)'.format(np.exp(Coefs[0].item()), Coefs[1].item()))
        plt.show()
    return Coefs,Yaprox  #Devuelve los coeficientes y los valores aproximados
###################################
###################################
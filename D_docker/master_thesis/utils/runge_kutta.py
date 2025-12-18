import numpy as np
import matplotlib.pyplot as plt
def runge_kutta_4_simulation(dynamic, x0, dt, u,t, plot=False, annotate = False, decimate_annotation= 1):
    # Calculate the number of time steps
    t_interp = np.arange(t[0], t[-1], dt)
    # print(t_interp)
    N = len(t_interp)

    # Create the time array
    # t = np.arange(0, T, dt)

    # Initialize the state variable and storage arrays
    x = np.zeros((x0.shape[0],N))
    x[:,[0]] = x0.reshape(-1,1)
    
    

    # Interpolate the input vector
    # print(t.shape,u.shape)
    # print(t_interp.shape,u.shape,t.shape)
   
    # print(u.shape)
    if u is list:
        u = np.array(u)
    if len(u.shape) == 1:
        u_interp = np.zeros((1,N))
        u_interp[[0],:] = np.interp(t_interp, t, u.squeeze())
    else:
        u_interp = np.zeros((u.shape[0],N))
        for i in range(u.shape[0]):
            u_interp[[i],:] = np.interp(t_interp, t, u[i,:])
    # print(u_interp)
    for i in range(N-1):
        # print(x[:,i],u_interp[:,i])
        k1 = np.array(dynamic(x[:,[i]], u_interp[:,i])).reshape(-1,1)
        k2 = np.array(dynamic(x[:,[i]] + k1 * dt/2, u_interp[:,[i]]/2 + u_interp[:,[i+1]]/2)).reshape(-1,1)
        k3 = np.array(dynamic(x[:,[i]] + k2 * dt/2, u_interp[:,[i]]/2 + u_interp[:,[i+1]]/2)).reshape(-1,1)
        k4 = np.array(dynamic(x[:,[i]] + k3 * dt, u_interp[:,[i+1]])).reshape(-1,1)
        # print(k2)
        # print(x[:,[i]])
        x[:,[i+1]] = x[:,[i]] + (k1 + 2*k2 + 2*k3 + k4) * dt/6

    # Mark and display the specific points
    if plot:

        plt.figure()
        plt.plot(t_interp, x.T, label='Simulation')
        plt.plot(t_interp, u_interp.T, label='Input')
        if annotate:
            point_times = t
            point_values = np.array([x[:,[int((tt-t[0]) / dt)]] for tt in t[0:-1]] + [x[:,[-1]]]).reshape(1,-1)
            for j in range(0,point_values.shape[0]):
                plt.scatter(point_times, point_values[j,:], color='red', label='Marked Points')
                for i, (t_point, x_point) in enumerate(zip(point_times, point_values[j,:])):
                    if i % decimate_annotation == 0:
                        plt.annotate(f'({t_point:.2f}, {x_point:.2f})', (t_point, x_point),
                                    textcoords="offset points", xytext=(0, 10), ha='center',size=8)

        plt.xlabel('Time (s)')
        plt.ylabel('x')
        plt.title('Dynamical Simulation')
        plt.legend()
        plt.show()
    return t_interp, x, u_interp

def runge_kutta_4_feedback_simulation(dynamic, x0,num_inputs,controller, dt,T, plot=False):
    # Calculate the number of time steps
    t_interp = np.arange(0, T, dt)
    
    # print(t_interp)
    N = len(t_interp)

    # Create the time array
    # t = np.arange(0, T, dt)

    # Initialize the state variable and storage arrays
    x = np.zeros((x0.shape[0],N))
    u = np.zeros((num_inputs,N))
    x[:,0] = x0.reshape(-1)
    
    
    # print(u_interp)
    for i in range(N-1):
        # print(x[:,i],u_interp[:,i])
        # print(x[:,[i]],t_interp[i])
        u[:,[i]] = controller(x[:,[i]],t_interp[i])
        k1 = np.array(dynamic(x[:,[i]], u[:,i])).reshape(-1,1)
        k2 = np.array(dynamic(x[:,[i]] + k1 * dt/2, u[:,[i]])).reshape(-1,1)
        k3 = np.array(dynamic(x[:,[i]] + k2 * dt/2, u[:,[i]])).reshape(-1,1)
        k4 = np.array(dynamic(x[:,[i]] + k3 * dt, u[:,[i]])).reshape(-1,1)
        # print(k2)
        # print(x[:,[i]])
        x[:,[i+1]] = x[:,[i]] + (k1 + 2*k2 + 2*k3 + k4) * dt/6
    u[:,[i+1]] = controller(x[:,[i+1]],t_interp[i+1])
    # Mark and display the specific points
    if plot:

        plt.figure()
        plt.plot(t_interp, x.T, label='Simulation')
        plt.plot(t_interp, u.T, label='Input')
        plt.xlabel('Time (s)')
        plt.ylabel('x')
        plt.title('Dynamical Simulation')
        plt.legend()
        plt.show()
    return t_interp, x, u

def rk4_step_control(x, u, f, dt):
    
    k1 = f(x, u)
    k2 = f(x + 0.5 * k1 * dt, u)
    k3 = f(x + 0.5 * k2 * dt, u)
    k4 = f(x + k3 * dt, u)
    
    # Combine the slopes to get the final state
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return x_next
def rk4_step(x, f, dt):
    
    k1 = f(x)
    k2 = f(x + 0.5 * k1 * dt)
    k3 = f(x + 0.5 * k2 * dt)
    k4 = f(x + k3 * dt)
    
    # Combine the slopes to get the final state
    x_next = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    return x_next

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("Controller Simulation")
    t,x,u = runge_kutta_4_feedback_simulation(lambda x,u: -x[0] + u[0], np.array([0]),1,lambda x,t: x - (x-1), T = 4,dt = 0.01, plot=False)
    plt.plot(t,x.T)
    plt.plot(t,u.T)
    plt.figure()
    print("Trajectory Simulation")
    # (dynamic, x0, dt, u,t, plot=False, annotate = False, decimate_annotation= 5)
    runge_kutta_4_simulation(lambda x,u: -x[0] + u[0], np.array([0]), dt = 0.01, u = np.array([1,2,3,0,0,0,0,0]), t = [1,2,3,4,5,6,7,8], plot=True, annotate=True, decimate_annotation= 1)
    plt.plot(t,x.T)
    plt.plot(t,u.T)
import numpy as np
import casadi as ca
import numba

# https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node2.html
# https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
class BSpline():
    order: int
    degree: int
    number_of_knots: int
    knots: np.ndarray
    control_points: np.ndarray
    def __init__(self, control_points, order):
        if type(control_points) is not np.ndarray and not isinstance(control_points, (ca.MX, ca.SX)):
            control_points = np.array(control_points)
        if len(control_points.shape) > 1:
            self.number_of_positions = control_points.shape[0]
            self.number_of_control_points = control_points.shape[1]
        else:
            self.number_of_positions = 1
            self.number_of_control_points = control_points.shape[0]
            control_points = control_points.reshape(1, -1)
        self.order = order
        self.degree = order - 1
        self.number_of_knots = self.number_of_control_points + order
        self.knots = np.array([0.0] * (order-1) + np.linspace(0.0, 1.0,
                              self.number_of_knots + 2 - order*2).tolist() + [1.0] * (order-1))
        self.control_points = control_points

    def evaluate(self, t, control_points=None):
        if control_points is None:
            control_points = self.control_points
        degree = self.degree
        knots = self.knots
        order = self.order
        #
        # if t <= 1e-6:
        #     t = 1e-6
        if t >= 1:
            t = 1-1e-16
        assert t >= 0 and t <= 1, f"t = {t}, must be in [0,1]"
        
        if control_points.shape[1] <= degree:
            return control_points[:, 0]

        # Find the index of the interval containing t
        index = 0
        while not (t >= knots[index] and t < knots[index + 1]):
            index += 1

        # Initialize a list with an 'order' number of slots
        points = [control_points[:, 0]] * order

        # Copy the active control points which go from (index - degree) to (index)
        for i in range(0, order):
            points[i] = control_points[:, index - degree + i]

        # Generate the intermediary control points, from lowest order to highest
        for l in range(0, degree):
            for j in range(degree, l, -1):
                alpha = (t - knots[index - degree + j]) / \
                    (knots[index - l + j] - knots[index - degree + j])
                # print(alpha)
                if alpha >= 1-1e-5:
                    points[j] = alpha * points[j]
                else:
                    points[j] = (1.0 - alpha) * points[j-1] + alpha * points[j]

        return points[degree]

    def create_derivative_spline(self):

        p = self.degree
        n = self.control_points.shape[1]
        knots = self.knots
        control_points = self.control_points
        Q = []
        for i in range(0, n-1):
            Q.append(p/(knots[i+p + 1]-knots[i+1]) *
                        (control_points[:, i+1]-control_points[:, i]))
            
        if isinstance(control_points, (ca.MX, ca.SX, ca.DM)):
            Q = ca.horzcat(*Q)
        else:
            Q = np.array(Q).T
        return BSpline(control_points=Q, order=self.order-1)
    @staticmethod
    @numba.njit
    def fast_create_derivative_spline_(degree,n,knots,control_points):

        p = degree
        Q = np.empty((control_points.shape[0], n-1))
        for i in range(0, n-1):
            Q[:, i] = p/(knots[i+p + 1]-knots[i+1]) * (control_points[:, i+1]-control_points[:, i])
        return Q
    def fast_create_derivative_spline(self):
        Q = self.fast_create_derivative_spline_(self.degree,self.control_points.shape[1],self.knots,self.control_points)
        return BSpline(control_points = Q, order=self.order-1)
    @staticmethod
    @numba.njit(parallel=True)
    def _fast_batch_evaluate(times,control_points,knots,order):
        # control_points = self.control_points
        # knots = self.knots
        # order = self.order
        out = np.empty((times.size,control_points.shape[0]))
        degree = order - 1
        for jj in numba.prange(times.size):
            t = times[jj]
            if t >= 1:
                t = 1-1e-16
            # assert t >= 0 and t <= 1, f"t = {t}, must be in [0,1]"
            
            if control_points.shape[1] <= degree:
                out[jj] = control_points[:, 0]

            # Find the index of the interval containing t
            index = 0
            while not (t >= knots[index] and t < knots[index + 1]):
                index += 1

            # Initialize a list with an 'order' number of slots
            points = [control_points[:, 0]] * order

            # Copy the active control points which go from (index - degree) to (index)
            for i in range(0, order):
                points[i] = control_points[:, index - degree + i]

            # Generate the intermediary control points, from lowest order to highest
            for l in range(0, degree):
                for j in range(degree, l, -1):
                    alpha = (t - knots[index - degree + j]) / \
                        (knots[index - l + j] - knots[index - degree + j])
                    # print(alpha)
                    if alpha >= 1-1e-5:
                        points[j] = alpha * points[j]
                    else:
                        points[j] = (1.0 - alpha) * points[j-1] + alpha * points[j]

            out[jj] = points[degree]
        return out
    def fast_batch_evaluate(self,times):
        return self._fast_batch_evaluate(times,self.control_points,self.knots,self.order)
    @staticmethod
    def fit_b_spline_to_points(num_control_points,order,points):
        if isinstance(points,(ca.MX,ca.SX,ca.DM)):
            points = points.full()
        if isinstance(points,(list,tuple)):
            points = np.array(points)
            
        assert isinstance(points,np.ndarray)
        number_of_positions = points.shape[0]
        number_of_samples = points.shape[1]
        opti = ca.Opti()
        control_points = opti.variable(number_of_positions,num_control_points)
        spline_sol = BSpline(control_points,order=order)
        spline_samples = ca.horzcat(*[spline_sol.evaluate(i) for i in np.linspace(0,1,number_of_samples)])
        obj = ca.sumsqr(points-spline_samples)
        opti.minimize(obj)
        opti.solver('ipopt',{'ipopt':{'print_level':0},'print_time':False})
        sol = opti.solve()
        return BSpline(sol.value(control_points),order=order)

    def record_in_meshcat(self,meshcat,diagram,plant,dt = 10,num_samples = 100,diagram_context = None):
        """
        Records (i.e, sends to meshcat) the trajectory
        """
        meshcat.StartRecording(frames_per_second=30.0)
        self.publish(diagram,plant,T = 0,dt = dt,num_samples = num_samples, diagram_context = diagram_context)
        meshcat.StopRecording()
        meshcat.PublishRecording()
        
    def publish(self,diagram,plant,T = 0,dt = 10,num_samples = 100, diagram_context = None):
        """
        Does `diagram.ForcedPublish(diagram_context)` where the positions of the plant is the evaluation of the BSpline at `i in np.linspace(0,1,num_samples)` sample.
        """
        if diagram_context is None:
            diagram_context = diagram.CreateDefaultContext()
            
        plant_context = plant.GetMyContextFromRoot(diagram_context)
        for i in np.linspace(0,1,num_samples):
            plant.SetPositions(plant_context, self.evaluate(i))
            diagram_context.SetTime(i*dt + T)
            diagram.ForcedPublish(diagram_context)
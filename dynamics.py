
# %%
import torch
from scipy.integrate import quad
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%

def D(x):
	return np.exp(-x**2/2)/(np.sqrt(2*np.pi))

class Model(nn.Module):

	def __init__(self, D_in, H, D_out):
		super(Model,self).__init__()

		self.w1 = torch.randn(H,D_in, device=device, dtype=dtype,requires_grad=True)
		self.w2 = torch.randn( D_out,H, device=device, dtype=dtype,requires_grad=True)
		self.gamma = torch.randn(H, D_in, device=device, dtype=dtype,requires_grad=True)

	def forward(self, x):

		h = torch.mm(self.w1,x).add_(self.gamma)

		h_relu = h.clamp(min=0)
		y_pred = torch.mm(self.w2,h_relu)
		return y_pred

	def hidden_cov(self,x):
		h = torch.mm(self.w1,x).add_(self.gamma)
		h_relu = h.clamp(min=0)
		cov = torch.mm(h_relu, h_relu.transpose(0,1))
		return cov

	def integrand_C1(self, x,index):

		w1_np=self.w1.detach().numpy()
		gamma_np = self.gamma.detach().numpy()
		out = D(x)*(w1_np[index]*x+gamma_np[index])*x
		return out

	def integrand_C2(self, x,index1, index2):

		w1_np=self.w1.detach().numpy()
		gamma_np = self.gamma.detach().numpy()
		out = D(x)*(w1_np[index1]*x+gamma_np[index1])*(w1_np[index2]*x+gamma_np[index2])
		return out

	def integrand_C_gamma(self, x, index):
		w1_np=self.w1.detach().numpy()
		gamma_np = self.gamma.detach().numpy()
		out = D(x)*(w1_np[index]*x+gamma_np[index])
		return out


dtype = torch.float
device = torch.device("cpu")


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 100, 1, 4, 1 

P_test = 20

sigma = .1 #level of noise


# Create random input and output data
x = torch.randn( D_in,N, device=device, dtype=dtype, requires_grad=True)
x_test = torch.randn( D_in,P_test, device=device, dtype=dtype, requires_grad=True)


epsilon = sigma*torch.randn(D_out,N, device=device, dtype=dtype)
epsilon_test = sigma*torch.randn(D_out,P_test, device=device, dtype=dtype)

w1_teach = torch.randn(H,D_in, device=device, dtype=dtype,requires_grad=True)
w2_teach = torch.randn( D_out,H, device=device, dtype=dtype,requires_grad=True)
gamma_teach = torch.randn(H, D_in, device=device, dtype=dtype,requires_grad=True)

h_teach = torch.mm(w1_teach,x).add_(gamma_teach)
h_relu_teach = h_teach.clamp(min=0)
y_noise_free = torch.mm(w2_teach,h_relu_teach)
y = y_noise_free.add_(epsilon)


h_teach_test = torch.mm(w1_teach,x_test).add_(gamma_teach)
h_relu_teach_test = h_teach_test.clamp(min=0)
y_noise_free_test = torch.mm(w2_teach,h_relu_teach_test)
y_test = y_noise_free_test.add_(epsilon_test)


num_epochs = 1000

learning_rate = 1e-4


#TODO: model for teacher, eign values of covariance matrix for student to teacher
model = Model(D_in, H, D_out)
teach = Model(D_in,H,D_out)



optimizer = torch.optim.SGD([model.w1,model.w2,model.gamma],lr=learning_rate)

for n in range(num_epochs):
	# Forward pass: compute predicted y
	y_pred = model(x)
	w1_np_prev=model.w1.detach().numpy()
	#print(model.w1, model.w2, model.gamma)
	loss = (1/N)*(y_pred - y).pow(2).sum() #.item() # == .sum() in numpy
	optimizer.zero_grad()
	loss.backward(retain_graph=True)
	optimizer.step()
	#print(model.w1, model.w2, model.gamma)
	w1_np=model.w1.detach().numpy()


	y_pred_test = model(x_test)
	mse_test = (1/P_test)*(y_pred_test - y_test).pow(2).sum()

	#print(loss.data.numpy())


	#print("generalization error: %f" %(mse_test))

	#print(model.hidden_cov(x_test))

	r = model.gamma/model.w1 # define separate for r_i and r_j
	r_teach = gamma_teach/w1_teach
	r_np = r.detach().numpy()
	r_teach_np = r_teach.detach().numpy() 
	r_inf = 10**5


	numX = 1000
	
	
	C1 =np.zeros((H,H))
	C1_teach = np.zeros((H,H))
	C2 =np.zeros((H,H))
	C2_teach = np.zeros((H,H))
	Cgamma =np.zeros((H,H))
	Cgamma_teach = np.zeros((H,H))

	for i in range(0,H):
		for j in range(0,H):

			r_curr = max(r_np[i], r_np[j])
			x_set = np.linspace(r_curr,r_inf,num=numX)
			dx = (r_inf-r_curr)/numX
			C1[i,j] = np.sum(model.integrand_C1(x_set,j))*dx

			r_curr_teach = max(r_teach_np[i], r_teach_np[j])
			x_set_teach = np.linspace(r_curr_teach,r_inf,num=numX)
			dx_teach = (r_inf-r_curr_teach)/numX
			C1_teach[i,j] = np.sum(model.integrand_C1(x_set_teach,j))*dx_teach


                        # C2
			
			gamma_np = model.gamma.detach().numpy()
			C2[i,j] = np.sum(model.integrand_C2(x_set,i,j))*dx


                        # C2_teach
			w1_teach_np=w1_teach.detach().numpy()
			gamma_teach_np = gamma_teach.detach().numpy()
			C2_teach[i,j] = np.sum(model.integrand_C2(x_set_teach,i,j))*dx_teach


			# gamma
			Cgamma[i,j] = np.sum(model.integrand_C_gamma(x_set,j))*dx

			Cgamma_teach[i,j] = np.sum(model.integrand_C_gamma(x_set_teach,j))*dx_teach


	#print(C1)
	w1_der_thy = np.zeros((1,H))
	w1_num_der = np.zeros((1,H))
	w2_der_thy = np.zeros((1,H))
	w2_num_der = np.zeros((1,H))

	w2_np = model.w2.detach().numpy()
	#print(np.dot(w2_np,C1[1,:].reshape(H,1))[0][0])

	for i in range(0,H):	
		w2_np = model.w2.detach().numpy()
#		w1_der_thy[i] = 
		C1_gap = C1_teach[i,:].reshape(H,1) - C1[i,:].reshape(H,1)
		#print(C1_gap)
		w1_num_der[0,i]=learning_rate*w2_np[0,i]*np.matmul(w2_np,C1_gap)
		#print(w1_num_der[0,i])
		w2_num_der[0,i]=learning_rate*np.matmul(w2_np,C1_gap)
		#print(w2_num_der[0,i])
		
		print(w1_np[i]-w1_np_prev[i])
		#print(w1_num_der[0,i])

# hidden layer dynamics


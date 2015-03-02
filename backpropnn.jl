###
# Input:
#    Layers - a vector of integers specifying the number of nodes at each
#     layer, i.e for all i, Layers(i) = number of nodes at layer i, there
#     must be at least three layers and the input layer Layers(1) must
#     equal the dimension of each vector in Input, likewise, Layers(end) 
#     must be equal to the dimension of each vector in Desired
#     N - training rate for network learning [0.1 - 0.9]
#     M - momentum for the weight update rule [0.0 - 0.9]
#     SatisfactoryMSE - the mse at which to terminate computation
#     Input - the training samples, a P-by-N matrix, where each Input[p] is
#      a training vector
#     Desired - the desired outputs, a P-by-M matrix where each Desired[p]
#      is the desired output for the corresponding input Input[p]
#
# Original Author: Dale Patterson
#  Version: 3.1.2
#  Date: 2.25.06
###

#(Layers,N,M,SatisfactoryMSE,Input,Desired)
function backprop(L,n,m,smse,X,Y)

	(P, N) = size(X)
	Pd = length(Y)
	M = size(Y,2)

	#initialization phase
	nLayers = length(L)

	w = cell(nLayers-1)

	for i=1:nLayers-2
		w[i] = [1 - 2.*rand(L[i+1],L[i]+1) ; zeros(1,L[i]+1)];
	end
	w[end] = 1 - 2.*rand(L[end],L[end-1]+1);

	mse = Inf
	epochs = 0
	#Pre-allocation phase
	#alphas
	a = cell(nLayers)
	a[1] = [X ones(P)]

	for i=2:nLayers-1
		a[i] = ones(P, L[i] + 1)
	end
	a[end] = ones(P, L[end])

	#println(a)

	#nets
	net = cell(nLayers-1)
	for i=1:nLayers-2
		net[i] = ones(P, L[i+1]+1)
	end
	net[end] = ones(P, L[end])

	#println(net)
	#delta weights
	prev_dw = cell(nLayers-1)
	sum_dw = cell(nLayers-1)

	for i=1:nLayers-1
		prev_dw[i] = zeros(w[i])
		sum_dw[i] = zeros(w[i])
	end
	while mse > smse && epochs < 30000
		for i=1:nLayers-1
			net[i] = a[i] * w[i]'

			if i < nLayers-1
				a[i+1] = [2 ./ (1+exp(-net[i][:,1:end-1]))-1 ones(P,1)];
			else
				a[i+1] = 2 ./ (1 + exp(-net[i])) - 1;
			end
		end
		err = Y-a[end]
		sse = sum(sum(err.^2))

		delta = err .* (1+a[end]) .* (1-a[end])
		for i=nLayers-1:-1:1
			sum_dw[i] = n * delta' * a[i]
			if i > 1
				delta = (1+a[i]) .* (1-a[i]) .* (delta * w[i])
			end
		end
		for i=1:nLayers-1
			prev_dw[i] = (sum_dw[i] ./ P) + (m*prev_dw[i])
			w[i] = w[i] + prev_dw[i]
		end
		epochs += 1;
		mse = sse/(P*M)
	end
@printf("iterations: %d\n", epochs)
w
end

function predict(L, w, X)
	P = size(X,1)
	nLayers = length(w)+1
	
	a = cell(nLayers)
	a[1] = [X ones(P)]
	for i=2:nLayers-1
		a[i] = ones(P, L[i] + 1)
	end
	a[end] = ones(P, L[end])

	net = cell(nLayers-1)
	for i=1:nLayers-2
		net[i] = ones(P, L[i+1]+1)
	end
	net[end] = ones(P, L[end])

	for i=1:nLayers-1
		net[i] = a[i] * w[i]'

		if i < nLayers-1
			a[i+1] = [2 ./ (1+exp(-net[i][:,1:end-1]))-1 ones(P,1)];
		else
			a[i+1] = 2 ./ (1 + exp(-net[i])) - 1;
		end
	end
	#err = Y-a[end]
	a
end
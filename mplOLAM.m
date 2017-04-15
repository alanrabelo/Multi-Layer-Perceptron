clc;
clear all;
close all;

inputs = importdata('iris.data');
% XOR input for x1 and x2

X = inputs(:, 1:4).';
X = normalize(X);
Y = inputs(:, 5:7).';


randomIndexes = randperm(length(X));

X_treino = X(:, (randomIndexes(1:120)));
Y_treino = Y(:, (randomIndexes(1:120)));
X_teste = X(:, (randomIndexes(121:150)));
Y_teste = Y(:, (randomIndexes(:, 121:150)));

quantidadeDeNeuroniosOculta = 4;
quantidadeDeAtributos = 4;
quantidadeDeNeuroniosSaida = 3;

Wh = rand(quantidadeDeAtributos + 1, quantidadeDeNeuroniosOculta);
Ws = rand(quantidadeDeNeuroniosOculta + 1, quantidadeDeNeuroniosSaida);

% APRESENTAR AS ENTRADAS A CADA NEURONIO DA CAMADA ESCONDIDA
xh = [repmat(-1, 1, 120); X_treino];
uh = Wh' * xh;
yh = arrayfun(@(uh) sigma(uh), uh);

xs = [repmat(-1, 1, 120); yh];
us = Ws' * xs;
ys = arrayfun(@(us) sigma(us), us);
ys = getOutput(ys);

erroSaida = Y_treino - ys;

inversa = ((xs' * xs)\xs');

Ws =Y_treino * inversa;

deltaWh = (erroSaida .* 0.1 * X_treino.')' * Ws;

  eitaman =  (1-(ys.^2));
  
Wh = Wh + deltaWh';


function y = sigma( x )
    y =1 / (1 + exp( -x ));
end

function y = getOutput(x)

y = zeros(size(x));
for j = 1:size(x, 2)
    max = 0;
    maxIndex = 1;
    for i = 1:size(x, 1)
        if x(i) > max
            max = x(i);
            maxIndex = i;
        end
    end
    
    y(maxIndex, j) = 1;
end

end


function [y] = normalize(x)

    y = (x - min(x)) ./ ( max(x) - min(x));
end

function y = sinalDe(x)
if x > 0.5
    y = 1;
else
    y = 0;
end
end

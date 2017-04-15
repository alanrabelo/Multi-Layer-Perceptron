clc;
clear all;
close all;

inputs = importdata('iris.data');
% XOR input for x1 and x2

entradas = inputs(:, 1:4);

entradas = (entradas - min(entradas)) ./ ( max(entradas) - min(entradas) );
saidasDesejadas = inputs(:, 5:7);

randIndexes = randperm(length(inputs));

X_treino = entradas(randIndexes(1:floor(length(entradas) * 0.8)), :)';
X_teste = entradas(randIndexes(floor(length(entradas) * 0.8) + 1:length(inputs)),:)';
Y_treino = saidasDesejadas(randIndexes(1:floor(length(saidasDesejadas) * 0.8)), :)';
Y_teste = saidasDesejadas(randIndexes(floor(length(saidasDesejadas) * 0.8) + 1:length(inputs)),:)';

X_treino = [repmat(-1, 1, 120); X_treino];
% Desired output of XOR
% saidasDesejadas = [0;1;1;0];
% Initialize the bias
% Learning coefficient
alpha = 0.5;

% Number of learning iterations
maximoDeEpocas = 10000;

%Numero de amostras de dados
quantidadeDeAmostras = size(X_treino, 2);

%Numero de entradas por amostra de dados
quantidadeDeEntradas = size(X_treino, 1);

quantidadeDeNeuroniosOcultos = 4;
quantidadeDeNeuroniosDeSaida = 3;

MSE = zeros(1,maximoDeEpocas);

convergence = 0;

while(convergence == 0)
    
    Woculto = rand(quantidadeDeEntradas, quantidadeDeNeuroniosOcultos);
    Wsaida = rand(quantidadeDeNeuroniosOcultos+1, quantidadeDeNeuroniosDeSaida);
    % pesos2 = zeros(quantidadeDeNeuroniosDeSaida, quantidadeDeNeuroniosOcultos+1);
    
    for i = 1:maximoDeEpocas
      
        for j = 1:quantidadeDeAmostras
            
            uCamadaOculta = Woculto' * X_treino(:,j);
            yCamadaOculta = logistica(uCamadaOculta);
            
            entradasDaCamadaDeSaida = [-1; yCamadaOculta];
            
            uDaCamadaDeSaida =  Wsaida' * entradasDaCamadaDeSaida;
            yCamadaSaida = logistica(uDaCamadaDeSaida);
            yCamadaSaidaDegrau = getOutput(yCamadaSaida);          
            % Testar com o degrau na saida
            erroNaSaida = saidasDesejadas(j, :)' - yCamadaSaidaDegrau;
%             logisticaRes = alpha * erroNaSaida * (yCamadaSaida - yCamadaSaida.^2)';
            logisticaRes = alpha * erroNaSaida;
            deltaCamadaSaida =  repmat(entradasDaCamadaDeSaida, 1, 3) * logisticaRes;
            Wsaida = Wsaida + deltaCamadaSaida;
            
            derivada =  alpha * erroNaSaida * (yCamadaSaida - yCamadaSaida.^2)';
            derivada2 = (Woculto * derivada);
            entradasDelta = repmat(yCamadaOculta, 1,4);
            deltaCamadaOculta =  entradasDelta * derivada2;
            Woculto = Woculto + deltaCamadaOculta;
            
        end
    end
    
   finalValues = Woculto' * X_treino;
   finalValues2 = Wsaida' * [repmat(-1,1,120); finalValues];
   finalDegrau = [];
   
   for i = 1:length(finalValues2)
       valor = finalValues2(:, i)
       degrau = getOutput(valor)
       finalDegrau = [finalDegrau degrau];
       
   end
   
   
        YFinal  = Wsaida*logistica(entradas*Woculto)'; % Sa�da da Rede;
        for i = 1:quantidadeDeAmostras
            if(YFinal(i) > 0.5)
                YFinal(i) = 1;
            else
                YFinal(i) = 0;
            end
        end
        E   = saidasDesejadas - YFinal;                    % Erros calculados;
        MSE(epoca) = ((E' * E)^0.5)/4;          % Erro Quadr�tico M�dio.

        % Crit�rio de parada [Erro M�nimo Alcan�ado]
        if MSE(epoca) < 0.001
            convergencia = 1;
            break 
        end
        
    disp('Deu erro')
    disp([sinalDe(saida(1)); sinalDe(saida(2));sinalDe(saida(3));sinalDe(saida(4))])
    
end

% for i = 1:maximoDeEpocas
%
%     saida = zeros(4,1);
%     numeroDeEntradas = length(entradas(:,1));
%
%     for j = 1:numeroDeEntradas
%

%

%

%

%         % Dessa forma podemos atualizar os pesos originais e repetir tudo
%         % de novo com os novos pesos
%         % varia??o dos pesos = taxa de aprendizado * x * delta
%
%         for k = 1:3
%             if k == 1 % Para o BIAS
%                 pesos(1,k) = pesos(1,k) + taxaDeAprendizado*entradas(j,1)*deltaCamadaOculta1;
%                 pesos(2,k) = pesos(2,k) + taxaDeAprendizado*entradas(j,1)*deltaCamadaOculta2;
%                 pesos(3,k) = pesos(3,k) + taxaDeAprendizado*entradas(j,1)*deltaCamadaDeSaida;
%             else % Para os outros casos
%                 pesos(1,k) = pesos(1,k) + taxaDeAprendizado*entradas(j,2)*deltaCamadaOculta1;
%                 pesos(2,k) = pesos(2,k) + taxaDeAprendizado*entradas(j,3)*deltaCamadaOculta2;
%                 pesos(3,k) = pesos(3,k) + taxaDeAprendizado*yDaCamadaEscondida(k-1)*deltaCamadaDeSaida;
%             end
%         end
%     end
%
%     if sinalDe(saida(1)) == saidasDesejadas(1) && sinalDe(saida(2)) == saidasDesejadas(2) && sinalDe(saida(3)) == saidasDesejadas(3) && sinalDe(saida(4)) == saidasDesejadas(4)
%        disp('Numero de epocas: ')
%        disp(i)
%        disp('Resultado Final:')
%        disp([sinalDe(saida(1)); sinalDe(saida(2));sinalDe(saida(3));sinalDe(saida(4))])
%         return
%
%     end
%
% end



function [ y ] = logistica( x )

for i = 1:size(x, 1)
    y(i, 1)=1/(1+exp(-x(i)));
end

end

function y = getOutput(x)

max = x(1);
maxIndex = 1;
for i = 1:size(x, 1)
    if x(i) > max
        max = x(i);
        maxIndex = i;
    end
end

y = zeros(size(x, 1), size(x, 2));
y(maxIndex) = 1;

end

function y = sinalDe(x)
if x >= 0
    y = 1;
else
    y = 0;
end
end

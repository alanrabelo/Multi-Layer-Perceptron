clc;
clear all;
close all;

inputs = importdata('iris.data');
% XOR input for x1 and x2

entradas = [-1 0 0; -1 0 1; -1 1 0; -1 1 1];
saidasDesejadas = [0; 1; 1; 0];


% Desired output of XOR
% saidasDesejadas = [0;1;1;0];
% Initialize the bias
% Learning coefficient
taxaDeAprendizado = 0.1;

% Number of learning iterations
maximoDeEpocas = 100000;

%Numero de amostras de dados
quantidadeDeAmostras = size(entradas, 1);

%Numero de entradas por amostra de dados
quantidadeDeEntradas = size(entradas, 2);

quantidadeDeNeuroniosOcultos = 2;
quantidadeDeNeuroniosDeSaida = 1;

saida = zeros(4,1);

MSE = zeros(1,maximoDeEpocas);

convergence = 0;

while(convergence == 0)
    
    pesos1 = (rand(quantidadeDeNeuroniosOcultos,quantidadeDeEntradas) - 0.5)/10;
pesos2 = (rand(quantidadeDeNeuroniosDeSaida,quantidadeDeNeuroniosOcultos+1) - 0.5)/10;
% pesos2 = zeros(quantidadeDeNeuroniosDeSaida, quantidadeDeNeuroniosOcultos+1);
saida = zeros(4,1);
    
    for i = 1:maximoDeEpocas
        disp(saida)
        
        saida = zeros(4,1);
        
        
        for j = 1:quantidadeDeAmostras
            
            uCamadaOculta = entradas(j,:) * pesos1.';
            yCamadaOculta = sigma(uCamadaOculta);
            
            entradasDaCamadaDeSaida = [-1 yCamadaOculta];
            
            uDaCamadaDeSaida = entradasDaCamadaDeSaida * pesos2.';
            yCamadaSaida = sigma(uDaCamadaDeSaida);
            saida(j) = sigma(uDaCamadaDeSaida);
            
            % Testar com o degrau na saida
            erroNaSaida = saidasDesejadas(j) - yCamadaSaida;
            
            deltaCamadaSaida = (sigma(yCamadaSaida) - sigma(yCamadaSaida).^2) * taxaDeAprendizado * erroNaSaida * entradasDaCamadaDeSaida;
            pesos2 = pesos2 + deltaCamadaSaida;
            
            derivada = (yCamadaOculta - yCamadaOculta.^2);
            deltaCamadaOculta = taxaDeAprendizado * erroNaSaida * repmat(pesos2(2:3),2,1) * derivada.' * entradas(j,:);
            pesos1 = pesos1 + deltaCamadaOculta;
            
            if sinalDe(saida(1)) == saidasDesejadas(1) && sinalDe(saida(2)) == saidasDesejadas(2) && sinalDe(saida(3)) == saidasDesejadas(3) && sinalDe(saida(4)) == saidasDesejadas(4)
                disp('Numero de epocas: ')
                disp(i)
                disp('Resultado Final:')
                disp([sinalDe(saida(1)); sinalDe(saida(2));sinalDe(saida(3));sinalDe(saida(4))])
                return
            end
            
        end
    end
    
        YFinal  = pesos2*tanh(entradas*pesos1)'; % Sa�da da Rede;
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



function [ y ] = sigma( x )

for i = 1:size(x, 2)
    y(i)=1/(1+exp(-x(i)));
end

end

function y = getOutput(x)

max = 0;
maxIndex = 1;
for i = 1:size(x, 2)
    if x(i) > max
        max = x(i);
        maxIndex = i;
    end
end

y = zeros(size(x, 1), size(x, 2));
y(maxIndex) = 1;

end

function y = sinalDe(x)
if x > 0.5
    y = 1;
else
    y = 0;
end
end

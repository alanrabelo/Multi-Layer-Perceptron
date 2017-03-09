
% XOR input for x1 and x2
entradas = [-1 0 0; -1 0 1; -1 1 0; -1 1 1];
% Desired output of XOR
saidasDesejadas = [0;1;1;0];
% Initialize the bias
% Learning coefficient
taxaDeAprendizado = 10;

taxaDeAprendizado = 0.5;
% Number of learning iterations
maximoDeEpocas = 100000;
pesos = zeros(3, 3);

for i = 1:maximoDeEpocas
    
    saida = zeros(4,1);
    numeroDeEntradas = length(entradas(:,1));
    
    for j = 1:numeroDeEntradas
        
        %%%% CAMADA ESCONDIDA
        
        % Aplicamos as entradas aos pesos e calculamos o u e o y (resultante da fun??o sigm?ide) do primeiro neur?nio
        uDaCamadaEscondida1 = pesos(1, :) * entradas(j, :).';
        yDaCamadaEscondida(1) = sigma(uDaCamadaEscondida1);
        
        % fazemos o mesmo para o segundo neur?nio
        uDaCamadaEscondida2 = pesos(2, :) * entradas(j, :).';
        yDaCamadaEscondida(2) = sigma(uDaCamadaEscondida2);
        
        %%%% CAMADA DE SA?DA
        hiddenInputs = [-1 yDaCamadaEscondida(1) yDaCamadaEscondida(2)];
        % Calculamos o u da camada de sa?da (no caso apenas um neur?nio) e seu respectivo y (sigm?ide)
        uDaCamadaDeSaida = pesos(3, :) * hiddenInputs.';
        saida(j) = sigma(uDaCamadaDeSaida);
        
        %%%% CALCULANDO O DELTA DOS PESOS
        % Na camada de sa?da calculamos:
        deltaCamadaDeSaida = saida(j)*(1-saida(j))*(saidasDesejadas(j)-saida(j));
        
        % Propagamos o erro para as camadas ocultas utilizando seu y
        deltaCamadaOculta1 = yDaCamadaEscondida(1)*(1-yDaCamadaEscondida(1))*pesos(3,2)*deltaCamadaDeSaida;
        deltaCamadaOculta2 = yDaCamadaEscondida(2)*(1-yDaCamadaEscondida(2))*pesos(3,3)*deltaCamadaDeSaida;
        
        % Dessa forma podemos atualizar os pesos originais e repetir tudo
        % de novo com os novos pesos
        % varia??o dos pesos = taxa de aprendizado * x * delta
        
        for k = 1:3
            if k == 1 % Para o BIAS
                pesos(1,k) = pesos(1,k) + taxaDeAprendizado*entradas(j,1)*deltaCamadaOculta1;
                pesos(2,k) = pesos(2,k) + taxaDeAprendizado*entradas(j,1)*deltaCamadaOculta2;
                pesos(3,k) = pesos(3,k) + taxaDeAprendizado*entradas(j,1)*deltaCamadaDeSaida;
            else % Para os outros casos
                pesos(1,k) = pesos(1,k) + taxaDeAprendizado*entradas(j,2)*deltaCamadaOculta1;
                pesos(2,k) = pesos(2,k) + taxaDeAprendizado*entradas(j,3)*deltaCamadaOculta2;
                pesos(3,k) = pesos(3,k) + taxaDeAprendizado*yDaCamadaEscondida(k-1)*deltaCamadaDeSaida;
            end
        end
    end
    
    if sinalDe(saida(1)) == saidasDesejadas(1) && sinalDe(saida(2)) == saidasDesejadas(2) && sinalDe(saida(3)) == saidasDesejadas(3) && sinalDe(saida(4)) == saidasDesejadas(4)
       disp('Numero de epocas: ') 
       disp(i) 
       disp('Resultado Final:')
       disp([sinalDe(saida(1)); sinalDe(saida(2));sinalDe(saida(3));sinalDe(saida(4))])       
        return
        
    end
    
end



function [ y ] = sigma( x )

    y=1/(1+exp(-x));

end

function y = sinalDe(x)
    if x > 0.5 
        y = 1;
    else 
        y = 0;
    end
end

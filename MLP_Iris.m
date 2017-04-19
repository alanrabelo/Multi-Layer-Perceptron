clc;
clear all;
close all;
%Carregando dados da iris embaralhados
B = load('iris.data');
B = B(randperm(size(B,1)),:);
%Separando amostras para treinamento e testes
treinamento = B(1:120,:);
teste = B(121:150,:);

nPadroes = 120;
nEntradas = 5;
%Definindo valores iniciais de pesos, taxa de apredizado e bias
bias = repmat(-1,nPadroes,1);

%X = randperm(X,1);
%Adicionando o bias a matriz de treinamento e testes
treinamento(:,1:4) = (treinamento(:,1:4) - repmat(min(treinamento(:,1:4)),120,1))./repmat((max(treinamento(:,1:4))-min(treinamento(:,1:4))),120,1);
teste(:,1:4) = (teste(:,1:4) - repmat(min(teste(:,1:4)),30,1))./repmat((max(teste(:,1:4))-min(teste(:,1:4))),30,1);
treinamento = [bias treinamento];
teste = [bias(1:30) teste];

erroEpoca = 0;
epoca = 0;
Y = zeros(nPadroes,3);
%Separando os valores de entradas X dos valores desejados D
D = treinamento(:,6:8);
X = treinamento(:,1:5);


neuronios_ocultos = size(X',2)/30;                    % Neurônios na camada oculta.
max_epocas        = 10000;                % Máximo de épocas permitidas.
nPadroes          = size(X,1);            % Número de Padrões;
nEntradas         = size(X,2);            % Dimensão do vetor de entrada. 

% Taxa de Aprendizado:

alpha    = 0.01;     % (camada oculta)
MSE = zeros(1,max_epocas); % Vetor para armazenar os erros de cada época.

convergencia = 0;          % Variável que atesta a convergência (0: FALSO)

while (convergencia == 0)
    
    % Geração dos pesos: há este laço mais externo para caso não haja con-
    % vergência, o processo seja reiniciado com novos pesos.
    
    w_oculta = (rand(nEntradas,neuronios_ocultos) - 0.5)/10;   
    w_saida  = (rand(3,neuronios_ocultos) - 0.5)/10;
    for epoca = 1:max_epocas
    
        % Laço para a varredura de todos os padrões.
        for j = 1:nPadroes

            x = X(j,:); % Selecionando o padrão.
            d = D(j,:); % Selecionando o valor de saída desejado.

            s_oculta = x*w_oculta;          % Ativação da camada oculta, 
            y_oculta = (tanh(s_oculta))';   % usandO uma função tanh(s).
            s_saida  = y_oculta'*w_saida';  % Ativação da camada de saída,
            y        = s_saida;             % usando uma função linear.
            erro     = d - y;               % Erro na saída.

            % Ajuste dos pesos da camada de saída:

            delta_w_saida  = erro'*alpha*y_oculta';
            w_saida        = w_saida + delta_w_saida;

            % Ajuste dos pesos da camada oculta:
            for i = 1:neuronios_ocultos
                delta_w_oculta = ...
                alpha*erro*w_saida(:,i)*(1-(y_oculta(i).^2))*x;
                w_oculta(:,i)       = w_oculta(:,i) + delta_w_oculta';
            end
        end
        
        % Fim da época. Ao fim da época, os pesos são testados para que se
        % conheça o erro quadrático médio da época.

        Y          = w_saida*tanh(X*w_oculta)'; % Saída da Rede;
        
        E          = D - Y';                    % Erros calculados;
        
        MSE(epoca) = sum(sum((E' * E)^0.5))/nPadroes;          % Erro Quadrático Médio.

        % Critério de parada [Erro Mínimo Alcançado]
        if MSE(epoca) < 0.001
            convergencia = 1;
            break 
        end

    end
end
%Iniciando os testes
Dteste = teste(:,6:8);
Xteste = teste(:,1:5);
erroTeste = 0;
YTeste = zeros(30,3);
for j = 1:30
    x = Xteste(j,:);
    d = Dteste(j,:);
    
    s_oculta = x*w_oculta;
    y_oculta = (tanh(s_oculta))';
    s_saida  = y_oculta'*w_saida';
    y        = s_saida;
    for i = 1:3
        if(y(:,i)> 0)
           YTeste(j,i) = 1;
        else
          YTeste(j,i) = 0;
        end
    end
    if(sum(YTeste(j,:))> 1)
       if y(1) > y(2) && y(1) > y(3)
           YTeste(j,:) = [1 0 0];
       elseif y(2) > y(3)
           YTeste(j,:) = [0 1 0];
       else
           YTeste(j,:) = [0 0 1];
       end
    end
    error(j,:) = Dteste(j,:)-YTeste(j,:);
    e = sum(abs(error(j,:)));
    if(e ~= 0)
        erroTeste = erroTeste + 1;
    end 
end
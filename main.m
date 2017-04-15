clc;
clear all;
close all;
%Carregando dados da iris embaralhados
dadosDoArquivo = load('iris.data');
dadosDoArquivo = dadosDoArquivo(randperm(size(dadosDoArquivo,1)),:);
%Separando amostras para treinamento e testes
treino = dadosDoArquivo(1:120,:);
teste = dadosDoArquivo(121:150,:);

numeroDeAmostras = length(treino);
numeroDeEntradas = 5;
%Definindo valores iniciais de pesos, taxa de apredizado e bias
bias = repmat(-1,numeroDeAmostras,1);

%X = randperm(X,1);
%Adicionando o bias a matriz de treinamento e testes
treino(:,1:4) = (treino(:,1:4) - repmat(min(treino(:,1:4)),120,1))./repmat((max(treino(:,1:4))-min(treino(:,1:4))),120,1);
teste(:,1:4) = (teste(:,1:4) - repmat(min(teste(:,1:4)),30,1))./repmat((max(teste(:,1:4))-min(teste(:,1:4))),30,1);
treino = [bias treino];
teste = [bias(1:30) teste];

erroEpoca = 0;
epoca = 0;
Y = zeros(numeroDeAmostras,3);
%Separando os valores de entradas X dos valores desejados D
D = treino(:,6:8);
X = treino(:,1:5);


neuroniosOcultos = size(X',2)/30;                    % Neurônios na camada oculta.
maximoDeEpocas        = 10000;                % Máximo de épocas permitidas.
numeroDeAmostras          = size(X,1);            % Número de Padrões;
numeroDeEntradas         = size(X,2);            % Dimensão do vetor de entrada. 

% Taxa de Aprendizado:

alpha    = 0.01;     % (camada oculta)
MSE = zeros(1,maximoDeEpocas); % Vetor para armazenar os erros de cada época.

convergencia = 0;          % Variável que atesta a convergência (0: FALSO)

while (convergencia == 0)
    
    % Geração dos pesos: há este laço mais externo para caso não haja con-
    % vergência, o processo seja reiniciado com novos pesos.
    
    pesosCamadaOculta = (rand(numeroDeEntradas,neuroniosOcultos) - 0.5)/10;   
    pesosCamadaSaida  = (rand(3,neuroniosOcultos) - 0.5)/10;
    for epoca = 1:maximoDeEpocas
    
        % Laço para a varredura de todos os padrões.
        for j = 1:numeroDeAmostras

            padraoAtual = X(j,:); % Selecionando o padrão.
            desejadoAtual = D(j,:); % Selecionando o valor de saída desejado.

            uOculta = padraoAtual*pesosCamadaOculta;          % Ativação da camada oculta, 
            yOculta = (tanh(uOculta))';   % usandO uma função tanh(s).
            uSaida  = yOculta'*pesosCamadaSaida';  % Ativação da camada de saída,
            ySaida        = uSaida;             % usando uma função linear.
            erro     = desejadoAtual - ySaida;               % Erro na saída.

            % Ajuste dos pesos da camada de saída:

            deltaWSaida  = erro'*alpha*yOculta';
            pesosCamadaSaida        = pesosCamadaSaida + deltaWSaida;

            % Ajuste dos pesos da camada oculta:
            for i = 1:neuroniosOcultos
                deltaWOculta = ...
                alpha*erro*pesosCamadaSaida(:,i)*(1-(yOculta(i).^2))*padraoAtual;
                pesosCamadaOculta(:,i)       = pesosCamadaOculta(:,i) + deltaWOculta';
            end
        end
        
        % Fim da época. Ao fim da época, os pesos são testados para que se
        % conheça o erro quadrático médio da época.

        Y          = pesosCamadaSaida*tanh(X*pesosCamadaOculta)'; % Saída da Rede;
        
        E          = D - Y';                    % Erros calculados;
        
        MSE(epoca) = sum(sum((E' * E)^0.5))/numeroDeAmostras;          % Erro Quadrático Médio.

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
    padraoAtual = Xteste(j,:);
    desejadoAtual = Dteste(j,:);
    
    uOculta = padraoAtual*pesosCamadaOculta;
    yOculta = (tanh(uOculta))';
    uSaida  = yOculta'*pesosCamadaSaida';
    ySaida        = uSaida;
    for i = 1:3
        if(ySaida(:,i)> 0)
           YTeste(j,i) = 1;
        else
          YTeste(j,i) = 0;
        end
    end
    if(sum(YTeste(j,:))> 1)
       if ySaida(1) > ySaida(2) && ySaida(1) > ySaida(3)
           YTeste(j,:) = [1 0 0];
       elseif ySaida(2) > ySaida(3)
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
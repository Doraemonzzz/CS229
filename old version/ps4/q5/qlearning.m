function [q, steps_per_episode] = qlearning(episodes)

% set up parameters and initialize q values
alpha = 0.05;
gamma = 0.99;
num_states = 100;
num_actions = 2;
actions = [-1, 1];
q = zeros(num_states, num_actions);
steps_per_episode = zeros(1, episodes);

for i=1:episodes,
  [x, s, absorb] =  mountain_car([0.0 -pi/6], 0);
  %%% YOUR CODE HERE
  %%% �ҵ���һ���Ķ�����Ӧ�����ֵ������
  [maxq, a] = max(q(s, :));
  %%% �����ͬ�����
  if q(s, 1) == q(s, 2)
      a = ceil(rand * num_actions);
  end
  %%% ���²���
  steps = 0;
  
  %%% ���δ����
  while(~absorb)
      %%% �ҵ���һ����λ��
      [x, sn, absorb] =  mountain_car(x, actions(a));
      %%% ����
      reward = - double(~ absorb);
      %%% �ҵ�������Ӧ�����ֵ������
      [maxq, an] = max(q(sn, :));
      %%% �����ͬ�����
      if q(s, 1) == q(s, 2)
        an = ceil(rand * num_actions);
      end
      %%% �ҵ������ж�
      q(s, a) = (1 - alpha) * q(s, a) + alpha * (reward + gamma * maxq);
      %%% ����״̬
      a = an;
      s = sn;
      steps = steps + 1;
  end
  %%% ��¼����
  steps_per_episode(i) = steps;
end
for i=1:10,
  [q, ep_steps] = qlearning(10000);
  all_ep_steps(i,:) = ep_steps;
end
plot(mean(reshape(mean(all_ep_steps), 500, 20)));
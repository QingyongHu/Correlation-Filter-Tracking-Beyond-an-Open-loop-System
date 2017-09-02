function pos = neighborhood_searching(Circle_center, radius, m, n)
% neighborhood area searching, determine a series of candidate position 
   delta_T = radius / m;
   delta_theta = 2 * pi / n;
   pos = Circle_center;
%    for ir  = 1:m
%        phase = mod(ir,2) * delta_theta / 2;
%        for it=1:n
%            dx = ir*delta_T*cos(it*delta_theta+phase);
%            dy = ir*delta_T*sin(it*delta_theta+phase);
%            
%            pos = [pos; [Circle_center(1) + dx, Circle_center(2) + dy]];
%        end
%    end

for i = 1:m
    for j = 1:n
        dx = i*delta_T*cos(j*delta_theta+((-1)^(i-1)+1)/4*delta_theta);
        dy = i*delta_T*sin(j*delta_theta+((-1)^(i-1)+1)/4*delta_theta);
        pos = [pos; [Circle_center(1) + dx, Circle_center(2) + dy]];
    end
end

end


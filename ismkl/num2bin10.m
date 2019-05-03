function [ binnum ] = num2bin10( num , N)
%num2bin10 Converts 0-9 to 10 digit fake binary
% num array of numbers


% if sum(num==0)
%     offset = 1;
%     N = N + 1;
% else
%     offset = 0;
% end


binnum = zeros(N, numel(num));
for i = 1:numel(num)
    binnum(num(i), i) = 1; 
end

end


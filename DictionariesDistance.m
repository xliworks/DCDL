function [ratio]=DictionariesDistance(original,new)

T=0.01;
Counter=0;
distances=abs(original'*new);
for i=1:size(original,2)
    minValue=1-max(distances(i,:));
    Counter=Counter+(minValue<T);
end;
ratio=100*Counter/size(original,2);
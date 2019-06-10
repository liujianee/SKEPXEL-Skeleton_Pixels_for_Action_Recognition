function [fourier_coeffs_lf] =  get_fourier_coeffs_pyramid(input, n_coeffs)

    n_levels = size(n_coeffs,2);
    
    L = size(input,2);


    fourier_coeffs_lf = [];
    for level = 1:n_levels
        st = floor(1:L/(2^(level-1)):L);
        en = [st(2:end)-1 , L];
        temp1 = [];
        for k = 1:length(st);
            coeffs = abs(fft(input(:,st(k):en(k)), n_coeffs(1,level),2));
            temp1 = [temp1, coeffs(:,1), 2*coeffs(:, 2:floor(n_coeffs(1,level)/4))];
        end
        fourier_coeffs_lf = [fourier_coeffs_lf, temp1];
    end
end

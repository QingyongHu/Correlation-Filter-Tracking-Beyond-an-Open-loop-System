function [positions, output_positions, time] = tracker(video_path, img_files, pos, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features, show_visualization,m_T,n_theta,num_init,response_threshold,PSR_threshold)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline for tracking with the KCF (by
%   choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%
%   It is meant to be called by the interface function RUN_TRACKER, which
%   sets up the parameters and loads the video information.
%
%   Parameters:
%     VIDEO_PATH is the location of the image files (must end with a slash
%      '/' or '\').
%     IMG_FILES is a cell array of image file names.
%     POS and TARGET_SZ are the initial position and size of the target
%      (both in format [rows, columns]).
%     PADDING is the additional tracked region, for context, relative to 
%      the target size.
%     KERNEL is a struct describing the kernel. The field TYPE must be one
%      of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%      POLY_A and POLY_B are the parameters for the Gaussian and Polynomial
%      kernels.
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%      target, relative to the target size.
%     INTERP_FACTOR is the adaptation rate of the tracker.
%     CELL_SIZE is the number of pixels per cell (must be 1 if using raw
%      pixels).
%     FEATURES is a struct describing the used features (see GET_FEATURES).
%     SHOW_VISUALIZATION will show an interactive video if set to true.
%
%   Outputs:
%    POSITIONS is an Nx2 matrix of target positions over time (in the
%     format [rows, columns]).
%    TIME is the tracker execution time, without video loading/rendering.
%
%   Joao F. Henriques, 2014
%   revised by: Qingyong Hu, April, 2017
%   https://github.com/QingyongHu

    sigma = 15;
    addpath('./utility');
    temp = load('w2crs');
    w2c = temp.w2crs;
	%if the target is large, lower the resolution, we don't need that much
	%detail
	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
	end
    resize_image2 = ((prod(target_sz) <= 750) && (prod(target_sz) >= 150));  %diagonal size >= threshold
    if resize_image2,
        interpolate_2 = 1.5;
        pos = floor(pos * interpolate_2);
        target_sz = floor(target_sz * interpolate_2);
    end
    
	%window size, taking padding into account
    if target_sz(1)/target_sz(2)> 2.5
        window_sz = floor(target_sz.*[2.2, 1+padding]);
    else
        window_sz = floor(target_sz * (1+padding));
    end

% 	%we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);

	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	

	
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
    output_positions = zeros(numel(img_files),4);
	positions = zeros(numel(img_files), 2);  %to calculate precision
    Peak_res = zeros(numel(img_files),1);
    currentScaleFactor =1;
   % Scale estimation
        nScales = 17;
        if nScales > 0
            scale_sigma_factor = 1/16;%1/4;
            nScalesInterp = 33;
            base_target_sz = target_sz;
            scale_step = 1.02;
            scale_model_factor = 1.0;
            init_target_sz = target_sz;
            scale_model_max_area = 512;
            learning_rate = 0.02;
            scale_sigma = nScalesInterp * scale_sigma_factor; % /sqrt(33)
            s_num_compressed_dim = 'MAX';
            scale_exp = (-floor((nScales-1)/2):ceil((nScales-1)/2)) * nScalesInterp/nScales;
            scale_exp_shift = circshift(scale_exp, [0 -floor((nScales-1)/2)]);

            interp_scale_exp = -floor((nScalesInterp-1)/2):ceil((nScalesInterp-1)/2);
            interp_scale_exp_shift = circshift(interp_scale_exp, [0 -floor((nScalesInterp-1)/2)]);

            scaleSizeFactors = scale_step .^ scale_exp;
            interpScaleFactors = scale_step .^ interp_scale_exp_shift;

            ys = exp(-0.5 * (scale_exp_shift.^2) /scale_sigma^2);
            ysf = single(fft(ys));
            scale_window = single(hann(size(ysf,2)))';

            %make sure the scale model is not to large, to save computation time
            if scale_model_factor^2 * prod(init_target_sz) > scale_model_max_area
                scale_model_factor = sqrt(scale_model_max_area/prod(init_target_sz));
            end

            %set the scale model size
            scale_model_sz = floor(init_target_sz * scale_model_factor);

            im = imread([video_path img_files{1}]);

            %force reasonable scale changes
            min_scale_factor = scale_step ^ ceil(log(max(5 ./ window_sz)) / log(scale_step));
            max_scale_factor = scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(scale_step));

            max_scale_dim = strcmp(s_num_compressed_dim,'MAX');
            if max_scale_dim
                s_num_compressed_dim = length(scaleSizeFactors);
            else
                s_num_compressed_dim = s_num_compressed_dim;
            end
        end
    
    
    
    for frame = 1:numel(img_files),
        %load image
        ita = 1;
        im = imread([video_path img_files{frame}]);
  
        if resize_image,
            im = imresize(im, 0.5);
        end
        if resize_image2,
            im = imresize(im, interpolate_2);
        end
        tic()
        
        if frame > 1,
            %obtain a subwindow for detection at the position from last
            %frame, and convert to Fourier domain (its size is unchanged)
            patch = get_translation_subwindow(im, pos, window_sz,currentScaleFactor);
            zf = fft2(get_features(patch, features, cell_size, cos_window,w2c));
            
            %calculate response of the classifier at all shifts
            switch kernel.type
                case 'gaussian',
                    kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                case 'polynomial',
                    kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                case 'linear',
                    kzf = linear_correlation(zf, model_xf);
            end
            response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            Peak_res(frame)  = max(response(:));  % peak value of the filter response map
            PSR_res = PSR(response,0.15);     % Initial Peak-to-Sidelobe Ratio (PSR) of the filter response map
            
            
            % Self-correction Mechanism
            if frame >= num_init
                temp_init = mean(Peak_res(3:num_init));
                
                % detect the abnormality in tracking output
                if (Peak_res(frame) < response_threshold && (temp_init > 0.35)) || (PSR_res  < PSR_threshold)
                    % neighborhood area searching, determine a series of candidate position
                    radius =  0.8 * (sqrt(0.025 / Peak_res(frame) * target_sz(1)^2 + 0.25 * target_sz(2)^2));
                    candidate_position = neighborhood_searching(pos,radius,m_T,n_theta);
                    responses_candidate = cell(length(candidate_position),1);
                    max_response_candidate = zeros(1,length(candidate_position));
                    
                    %calculate response of all candidate positions
                    for index_candidate = 1:length(candidate_position)
                        pos = candidate_position(index_candidate,:);
                        patch = get_subwindow(im, pos, window_sz);
                        zf = fft2(get_features(patch, features, cell_size, cos_window,w2c));
                        switch kernel.type
                            case 'gaussian',
                                kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                            case 'polynomial',
                                kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                            case 'linear',
                                kzf = linear_correlation(zf, model_xf);
                        end
                        responses_candidate{index_candidate} = real(ifft2(model_alphaf .* kzf));
                        max_response_candidate(index_candidate)=max(max(responses_candidate{index_candidate}));
                    end
                    
                    % minimizing the discrepancy between the tracking output and the expected response
                    index_patch = find(max_response_candidate == max(max_response_candidate), 1);
                    response = responses_candidate{index_patch};
                    pos = candidate_position(index_patch,:);
                    max_record = max(response(:));
                    if  max_record < 0.2
                        ita = exp((-(max_record-0.3).^2) /2 *sigma^2);
                    else
                        ita = 1;
                    end
                end
                
            end
            
            %target location is at the maximum response. we must take into
            %account the fact that, if the target doesn't move, the peak
            %will appear at the top-left corner, not at the center (this is
            %discussed in the paper). the responses wrap around cyclically.
            [vert_delta, horiz_delta] = find(response == max(response(:)), 1);
            if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
                vert_delta = vert_delta - size(zf,1);
            end
            if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
                horiz_delta = horiz_delta - size(zf,2);
            end
            pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
            
            if nScales > 0
                
                %create a new feature projection matrix
                [xs_pca, xs_npca] = get_scale_subwindow(im,pos,base_target_sz,currentScaleFactor*scaleSizeFactors,scale_model_sz);
                
                xs = feature_projection_scale(xs_npca,xs_pca,scale_basis,scale_window);
                xsf = fft(xs,[],2);
                
                scale_responsef = sum(sf_num .* xsf, 1) ./ (sf_den + lambda);
                
                interp_scale_response = ifft( resizeDFT(scale_responsef, nScalesInterp), 'symmetric');
                
                recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);
                
                %set the scale
                currentScaleFactor = currentScaleFactor * interpScaleFactors(recovered_scale_index);
                %adjust to make sure we are not to large or to small
                if currentScaleFactor < min_scale_factor
                    currentScaleFactor = min_scale_factor;
                elseif currentScaleFactor > max_scale_factor
                    currentScaleFactor = max_scale_factor;
                end
            end
        end
        
		%obtain a subwindow for training at newly estimated target position
        patch = get_translation_subwindow(im, pos, window_sz,currentScaleFactor);
		xf = fft2(get_features(patch, features, cell_size, cos_window,w2c));

		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
		switch kernel.type
		case 'gaussian',
			kf = gaussian_correlation(xf, xf, kernel.sigma);
		case 'polynomial',
			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
		case 'linear',
			kf = linear_correlation(xf, xf);
		end
		alphaf = yf ./ (kf + lambda);   %equation for fast training

        %Compute coefficents for the scale filter
        
        if nScales > 0
            
            %create a new feature projection matrix
            [xs_pca, xs_npca] = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz);
            if frame == 1
                s_num = xs_pca;
            else
                s_num = (1 - learning_rate) * s_num + learning_rate * xs_pca;
            end;
            bigY = s_num;
            bigY_den = xs_pca;
            
            if max_scale_dim
                [scale_basis, ~] = qr(bigY, 0);
                [scale_basis_den, ~] = qr(bigY_den, 0);
            else
                [U,~,~] = svd(bigY,'econ');
                scale_basis = U(:,1:s_num_compressed_dim);
            end
            scale_basis = scale_basis';
            
            %create the filter update coefficients
            sf_proj = fft(feature_projection_scale([],s_num,scale_basis,scale_window),[],2);
            sf_num = bsxfun(@times,ysf,conj(sf_proj));
            
            xs = feature_projection_scale(xs_npca,xs_pca,scale_basis_den',scale_window);
            xsf = fft(xs,[],2);
            new_sf_den = sum(xsf .* conj(xsf),1);
            if frame == 1
                sf_den = new_sf_den;
            else
                sf_den = (1 - learning_rate) * sf_den + learning_rate * new_sf_den;
            end;
        end
            
		if frame == 1,  %first frame, train with a single image
			model_alphaf = alphaf;
			model_xf = xf;
		else
			%subsequent frames, interpolate model
            interp_factor_current = interp_factor * ita;
            model_alphaf = (1 - interp_factor_current) * model_alphaf + interp_factor_current * alphaf;
            model_xf = (1 - interp_factor_current) * model_xf + interp_factor_current * xf;
		end
        target_sz = floor(base_target_sz * currentScaleFactor);

		%save position and timing
		positions(frame,:) = pos;
		time = time + toc();
        output_positions(frame,:) = [round(pos([2,1]) - target_sz([2,1])/2+0.0001), target_sz([2,1])];
        if resize_image,
            %output_positions(frame,:) = [round(pos([2,1])*2 - target_sz([2,1])), round(pos([2,1])*2 + target_sz([2,1]))];  %% princenton dataset
            output_positions(frame,:) = [round(pos([2,1])*2 - target_sz([2,1])),target_sz([2,1])*2];
        end

        %visualization
        if show_visualization,
            box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
            % 			stop = update_visualization(frame, box);
            if frame == 1,  %first frame, create GUI
                figure('IntegerHandle','off', 'Name',['Tracker - ' video_path]);
                im_handle = imshow(uint8(im), 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
                rect_handle = rectangle('Position',box, 'EdgeColor','g','LineWidth',3 );
                text_handle = text(10, 20, int2str(frame),'FontSize',24);
                set(text_handle, 'color', [1 1 0]);
            else
                try  %subsequent frames, update GUI
                    set(im_handle, 'CData', im)
                    set(rect_handle, 'Position', box)
                    set(text_handle, 'string', int2str(frame),'FontSize',24);
                catch
                    return
                end
            end
            drawnow
            % 			pause(0.05)  %uncomment to run slower
            
            
        end
        
        if resize_image,
            positions = positions * 2;
        end
    end
    if resize_image2,
        positions = positions / interpolate_2;
        output_positions = output_positions/interpolate_2;
    end
end


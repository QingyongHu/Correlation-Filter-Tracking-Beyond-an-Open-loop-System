function out = get_translation_subwindow(im, pos, model_sz, currentScaleFactor )
%GET_SUBWINDOW Obtain sub-window from image, with replication-padding.
%   Returns sub-window of image IM centered at POS ([y, x] coordinates),
%   with size SZ ([height, width]). If any pixels are outside of the image,
%   they will replicate the values at the borders.
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

	if isscalar(model_sz),  %square sub-window
		model_sz = [model_sz, model_sz];
    end
	
  
        %make sure the size is not to small
    if model_sz(1) < 1
        model_sz(1) = 2;
    end;
    if model_sz(2) < 1
        model_sz(2) = 2;
    end;

    sz = floor(model_sz * currentScaleFactor);
    
	xs = floor(pos(2)) + (1:sz(2)) - floor(sz(2)/2);
	ys = floor(pos(1)) + (1:sz(1)) - floor(sz(1)/2);
	
	%check for out-of-bounds coordinates, and set them to the values at
	%the borders
	xs(xs < 1) = 1;
	ys(ys < 1) = 1;
	xs(xs > size(im,2)) = size(im,2);
	ys(ys > size(im,1)) = size(im,1);
	
	%extract image
	im_patch = im(ys, xs, :);
    
    % resize image to model size
    out = mexResize(im_patch, model_sz, 'auto');

end


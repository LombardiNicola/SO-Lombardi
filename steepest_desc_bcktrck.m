function [xk, fk, gradfk_norm, kk, xseq, btseq] = steepest_desc_bcktrck(x0, f, gradf, alpha0, kmax, tolgrad, c1, rho, btmax)
    xk = x0;
    fk = f(xk);
    xseq = zeros(length(x0),kmax+1);
    btseq = zeros(1,kmax+1);
    xseq(:,1) = xk;
    for kk = 1:kmax
        pk = -gradf(xk);
        check = true;
        alpha = alpha0;
        t = 0;
        while check
            if t == btmax-1
                check = false;
            end
            xk = xseq(:, kk)+alpha*pk;
            temp = f(xk);
            if temp <= fk-c1*alpha*pk'*pk
                fk = temp;
                check = false;
            else
                alpha = rho*alpha;
                t = t+1;
            end
        end
        xseq(:, kk+1) = xk;
        btseq(kk+1) = t;
        gradfk_norm = norm(pk);
        if gradfk_norm < tolgrad
            break;
        end
    end
    xseq = xseq(:, 1:kk+1);
    btseq = btseq(1:kk+1);
end
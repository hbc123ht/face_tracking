class KalmanFilter{
    #_init
    constructor(dim_x, dim_z, dim_u = 0){
        if (dim_x < 1) console.log('dim_x must be 1 or greater');
        if (dim_z < 1) console.log('dim_z must be 1 or greater');
        if (dim_u < 0) console.log('dim_u must be 0 or greater');
        this.dim_x = dim_x;
        this.dim_z = dim_z;
        this.dim_u = dim_u;
        
        this.x = nj.zeros([dim_x, 1]);              //state 
        this.P = nj.identity(dim_x);                //uncertainty covariance
        // for (var i = 0;i < 7; i++)
        //     for (var j = 0;j < 7; j++) this.P.set(i, j, this.P.get(i, j) * 100);
        this.Q = nj.identity(dim_x);                //process uncertainty
        // for (var i = 0;i < 7; i++)
        //     for (var j = 0;j < 7; j++) this.Q.set(i, j, this.Q.get(i, j) * 10);
        this.B = null;                              //control transition matrix
        this.F = nj.identity(dim_x);                 //state transition matrix
        this.H = nj.zeros([dim_z, dim_x]);           //measurement function
        this.R = nj.identity(dim_z);                 //state uncertainty
        this._alpha_sq = 4.;                         //fading memory control 
        self.M = nj.zeros([dim_z, dim_z]);           //process-measurement cross correlation
        this.z = nj.array([[null] * this.dim_z]).T;
        // gain and residual are computed during the innovation step. We
        // save them so that in case you want to inspect them for various
        // purposes

        this.K = nj.zeros([dim_x, dim_z]);
        this.y = nj.zeros([dim_z, 1]);
        this.S =  nj.zeros([dim_z, dim_z]);
        this.SI = nj.zeros([dim_z, dim_z]);
        
        // identity matrix. Do not alter this.
        this._I = nj.identity(dim_x);
        this.duration = 0;

    }

    predict(u = null, B = null, F = null, Q = null){
        /*
        Predict next state (prior) using the Kalman filter state propagation
        equations.

        Parameters
        ----------

        u : np.array
            Optional control vector. If not `None`, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        */
        if (B == null){
            B = this.B;
        }
        if (F == null){
            F = this.F;
        }
        if (Q == null){
            Q = this.Q;
        }
        // Notice later

        if (B != null && u != null){
            this.x = nj.add(nj.dot(F, this.x), u);
        }
        else {
            this.x = nj.dot(F, this.x)
        }
        this.P = nj.dot(nj.dot(F, this.P), F.T);
        for (var i = 0;i < this.P.shape[0]; i++)
        for (var j = 0;j < this.P.shape[1]; j++) this.P.set(i, j, this.P.get(i, j) * this._alpha_sq);
    }

    update(z, R = null, H = null){
        /*
        Add a new measurement (z) to the Kalman filter.

        If z is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z is set to None.

        Parameters
        ----------
        z : (dim_z, 1): array_like
            measurement for this update. z can be a scalar if dim_z is 1,
            otherwise it must be convertible to a column vector.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise  self.+R will be used.

        H : np.array, or None
            Optionally provide H to override the measurement function for this
            one call, otherwise self.H will be used.
        */
        this._log_likelihood = null;
        this._likelihood = null;
        this._mahalanobis = null;
        //console.log(z);
        z = z.reshape([this.dim_z, 1]);

        if (R == null){
            R = this.R;
        }
        if (H == null){
            H = this.H;
        
        }
        //error between measurement and prediction
        this.y = nj.subtract(z, nj.dot(H, this.x));

        //common subexpression for speed
        var PHT = nj.dot(this.P, H.T);

        //project system uncertainty into measurement space
        this.S = nj.add(nj.dot(H, PHT), R);
        
        var tmp = [[this.S.get(0, 0), this.S.get(0, 1), this.S.get(0, 2), this.S.get(0, 3)],
                    [this.S.get(1, 0), this.S.get(1, 1), this.S.get(1, 2), this.S.get(1, 3)],
                    [this.S.get(2, 0), this.S.get(2, 1), this.S.get(2, 2), this.S.get(2, 3)],
                    [this.S.get(3, 0), this.S.get(3, 1), this.S.get(3, 2), this.S.get(3, 3)]];
        
        this.SI = nj.array(math.inv(tmp));
        //map system uncertainty into kalman gain
        this.K = nj.dot(PHT, this.SI);

        //predict new x with residual scaled by the kalman gain
        this.x = nj.add(this.x, nj.dot(this.K, this.y));

        var I_KH = nj.subtract(this._I, nj.dot(this.K, H));
        this.P = nj.add(nj.dot(nj.dot(I_KH, this.P), I_KH.T) , nj.dot(nj.dot(this.K, R), this.K.T));

        // this.P = nj.dot(I_KH, this.P);
        //save measurement and posterior state
        this.z = z.clone();
    }
}

function convert_x_to_bbox(x, score = null) {
    let w = Math.sqrt(x.get(2, 0) * x.get(3, 0));
    h = x.get(2, 0) / w;
    if (score == null){
        return nj.array([x.get(0, 0) - w/2., x.get(1, 0) - h/2., x.get(0, 0) + w/2.,x.get(1, 0)+h/2.]).reshape(4, 1);
    }
    else{
        return nj.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape(5, 1);
    }
}

function convert_bbox_to_z(bbox){
    let w = bbox.get(2, 0) - bbox.get(0, 0);
    let h = bbox.get(3, 0) - bbox.get(1, 0);
    let x = bbox.get(0, 0) + w/2.;
    let y = bbox.get(1, 0) + h/2.;
    let s = w * h;
    let r = w / h;
    return nj.array([x, y, s, r]).reshape(4, 1);
}

class KalmanBoxTracker{
    constructor(bbox){
        //define constant velocity model
        this.kf = new KalmanFilter(7, 4);
        this.kf.F = nj.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]);
        this.kf.H = nj.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]]);
        // this.kf.R[2:, 2:] *= 10
        for (var i = 2; i < this.kf.R.shape[0]; i++) for (var j = 2; j < this.kf.R.shape[1]; j++) this.kf.R.set(i, j, this.kf.R.get(i, j) * 10.);
            
        // this.kf.P.slice(4, 4) *= 100;
        for (var i = 4; i < this.kf.P.shape[0]; i++) for (var j = 4; j < this.kf.P.shape[1]; j++) this.kf.P.set(i, j, this.kf.P.get(i, j) * 1000.);
        for (var i = 0; i < this.kf.P.shape[0]; i++) for (var j = 0; j < this.kf.P.shape[1]; j++) this.kf.P.set(i, j, this.kf.P.get(i, j) * 10.);

        // this.kf.Q.slice(-1, -1) *= 0.01;
        this.kf.Q.set(this.kf.Q.shape[0] - 1, this.kf.Q.shape[1] - 1, this.kf.Q.get(this.kf.Q.shape[0] - 1, this.kf.Q.shape[1] - 1) * 0.01);

        // this.kf.Q.slice(4, 4) *= 0.01;
        for (var i = 4; i < this.kf.Q.shape[0]; i++) for (var j = 4; j < this.kf.Q.shape[1]; j++) this.kf.Q.set(i, j, this.kf.Q.get(i, j) * 0.01);
        
        // this.kf.x.slice(4) = convert_x_to_bbox(bbox);
        var tmp = convert_bbox_to_z(bbox);
        for (var i = 0; i < 4; i++) this.kf.x.set(i, 0, tmp.get(i,0));

        this.time_since_update = 0;

        this.history = [];
        this.hits = 0;
        this.hit_streak = 0;
    }

    update(bbox){
        this.time_since_update = 0;
        this.history = [];
        this.hits += 1;
        this.hit_streak += 1;
        this.kf.update(convert_bbox_to_z(bbox));
    }

    predict(){
        if (this.kf.x.get(6, 0) + this.kf.x.get(2, 0) <= 0){
            this.kf.x.set(6, 0, 0.0);
        }
        this.kf.predict();
        // console.log("this.kf : ", this.kf.x);
        if (this.time_since_update > 0){
            this.hit_streak = 0;
        }
        this.time_since_update += 1;
        return convert_x_to_bbox(this.kf.x);
    }

    get_state(){
        //return the current bounding box extimate.
        return convert_x_to_bbox(this.kf.x);
    }
}

function iou_score(boxA, boxB){
    boxA = nj.array(boxA);
    boxB = nj.array(boxB);
    boxA.reshape([4]);
    //console.log(boxB.shape);
    boxB.reshape([4]);

	var xA = math.max(boxA.get(0, 0), boxB.get(0, 0));
	var yA = math.max(boxA.get(1, 0), boxB.get(1, 0));
	var xB = math.min(boxA.get(2, 0), boxB.get(2, 0));
    var yB = math.min(boxA.get(3, 0), boxB.get(3, 0));
    
	var interArea = math.max(0, xB - xA + 1) * math.max(0, yB - yA + 1);
	var boxAArea = (boxA.get(2, 0) - boxA.get(0, 0) + 1) * (boxA.get(3, 0) - boxA.get(1, 0) + 1);
    var boxBArea = (boxB.get(2, 0) - boxB.get(0, 0) + 1) * (boxB.get(3, 0) - boxB.get(1, 0) + 1);
    
    var iou = interArea / (boxAArea + boxBArea - interArea);
	return iou;

}

class Sort{
    constructor(iou_thresold = 0.2){
        this.iou_thresold = iou_thresold;
        this.trackers = [];
        this.label = [];
        this.count = 0;
        this.num = [];
        this.id = []
    }
    update(new_pos){
        var predicts = [];
        var matchs = [];
        var re = [];
        re.length = new_pos.length;
        var exist = [];
        if (this.trackers.length > 0) exist.length = this.trackers.length;
            
        var pre = [];
        for (var i = 0;i < this.trackers.length; i++){
            
            predicts.push(this.trackers[i].predict());
        }      
        for (var i = 0;i < predicts.length; i++){
            for (var j = 0;j < new_pos.length; j++){
                var score = iou_score(predicts[i], new_pos[j]);
                if (score > this.iou_thresold){
                    matchs.push([score, i, j]);
                }
            }
        }
        matchs.sort(function(a, b){return b[0] - a[0]});
        for (var i = 0;i < matchs.length; i++){
            var x = matchs[i];
            if (re[x[2]] == null && exist[x[1]] == null){
                re[x[2]] = [new_pos[x[2]][0], new_pos[x[2]][1], new_pos[x[2]][2], new_pos[x[2]][3], this.id[x[1]]];
                var tmp = nj.array([new_pos[x[2]][0], new_pos[x[2]][1], new_pos[x[2]][2], new_pos[x[2]][3]]);
                tmp = tmp.reshape([4, 1]);
                this.trackers[x[1]].kf.duration = 0;
                this.trackers[x[1]].update(tmp);

                exist[x[1]] = 1;
            }
        }
        for (var i = 0;i < re.length; i++){
            if (x == null){
                this.count++;
                re[i] = [new_pos[i][0], new_pos[i][1], new_pos[i][2], new_pos[i][3], this.count];
                var tmp = nj.array([new_pos[i][0], new_pos[i][1], new_pos[i][2], new_pos[i][3]]);
                tmp.reshape([4, 1]);

                this.trackers.push(new KalmanBoxTracker(tmp));
                this.id.push(this.count);
            }
        }
        for (var i = 0;i < this.trackers.length; i++){
            if (exist[i] == null){
                console.log(this.trackers[i].kf.duration);
                this.trackers[i].kf.duration += 1;
                if (this.trackers[i].kf.duration > 30){
                    this.trackers.splice(i, 1);
                    this.id.splice(i, 1);
                }
                else {
                    var tmp = this.trackers[i].predict();
                    this.trackers[i].update(tmp);
                    re.push([tmp[0], tmp[1], tmp[2], tmp[3],0]);
                }
            }
        }

        return re;
    }

}
/*
var tracker;

for (var i = 0;i < 5; i++){
    var tmp = nj.zeros([5, 1]);
    for (var j = 0;j < lab.shape[1]; j++) {
        tmp.set(j, 0, lab.get(i, j));
    }
    if (i == 0) {tracker = new KalmanBoxTracker(tmp);}
    tracker.update(tmp);
    console.log("real : ", tmp);
    console.log("predict : ",tracker.predict());
}*/

lab = [
    [[286.552, 154.138, 357.889, 321.466],
[222.571, 179.989, 256.338, 288.916],
[456.542, 213.434, 481.675, 270.969],
[542.825, 206.921, 566.939, 251.902]],

[[294.638, 150.194,349.924, 328.68],
[220.799, 176.823, 255.684, 290.491],
[454.444, 205.312, 481.393, 268.44]],

[[300.521, 147.749, 351.412, 331.397],
[220.699, 180.031, 255.578, 294.856],
[453.872, 204.048, 480.685, 272.491]],

[[296.265, 145.238, 363.293, 338.047],
 [220.556, 183.745, 255.705, 294.633],
 [453.158, 210.629, 477.02, 268.975],
 [585.189, 211.552, 606.169, 246.765]],

 [[303.088, 145.696, 365.503, 333.434],
 [218.008, 177.205, 255.168, 301.563],
 [450.625, 214.432, 473.923, 266.693]],

 [[304.084, 137.272, 378.464, 337.122],
 [217.269, 183.92, 255.616, 292.802],
 [451.949, 215.578, 474.736, 268.932]]]


/*
 var face_trackers = new Sort();
 for (var i = 0;i < 6; i++){
     console.log("data : ",lab[i]);
    var label = face_trackers.update(lab[i]);
     console.log("result : ",label);
}
*/
/*
function associate_detection_to_trackers(detections, trackers, iou_thresold = 0.3){
    
    iou_matrix = iou_batch()
}*/
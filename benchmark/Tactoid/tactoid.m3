/* *****************************************************************
 * Surface height variation promoted by a nematic defect
 * ***************************************************************** */

m=manifoldload("Tactoid/disk.mesh")
m[1].orient()

/* *****************************************************************
 * Set up initial configuration
 * ***************************************************************** */

nn=m.newfield(multivector({0, {1,0,0}, {0,0,0}, 0}), coordinates={x,y,z})

/* *****************************************************************
 * Visualize a nematic configuration
 * ***************************************************************** */

mvectolist(mv) = { (mv.tolist())[2] }

visualize(m, n, dl) = {
	g=m.draw();
	indx=n.indices();
	do(
		g.cylinder(m[1].vertexposition(indx[i])-dl*mvectolist(n[indx[i]])/2,m[1].vertexposition(indx[i])+dl*mvectolist(n[indx[i]])/2);
		,{i,length(indx)}
	);
	g
};

@cmap = {
redcomponent(x) = 0.29+0.82*x-0.17*x^2;
greencomponent(x) = 0.04+1.66*x-0.79*x^2;
bluecomponent(x) = 0.54+1.18*x-0.9*x^2;
}

cm = new(cmap);

/* *****************************************************************
 * Set up energies and constraints
 * ***************************************************************** */

sigma = 5.0 * 0.04;
W= 4.0 * 0.4;
k11=0.2;
k33=0.2;

// Nematic energy
lf=new(nematic);
lf.setdirector(nn);
lf.setprefactor(1.0);
lf.setconstants(splay=k11, bend=k33);
m.addenergy(lf);

// Set up local director constraint
ln=new(fieldnormsq);
ln.setfield(nn);
ln.settarget(1.0); // Unit length
m.addlocalfieldconstraint(ln);

bnd = m[1].selectboundary();

// Apply a line tension to the boundary
lt=new(linetension);
m.addenergy(lt, bnd);
lt.setprefactor(sigma+W);

// Nematic anchoring
la=new(nematicanchoring)
la.setdirector(nn);
la.setprefactor(-W);
m.addenergy(la, bnd);

// Area constraint
laa=new(surfacetension)
m.addconstraint(laa)

// Equiarea
leq=new(equiareaweighted)

m.showfunctionals()

/* *****************************************************************
 * Renormalize
 * ***************************************************************** */

mvnorm(x) = sqrt((x*x)[0])

normalize(n) = {
	ii=n.indices();
	do( n[ii[i]] = n[ii[i]]*(1/mvnorm(n[ii[i]])) ,{i,length(ii)})
}

/* *****************************************************************
 * Minimization and other functions
 * ***************************************************************** */

minimizefield(niter, fsc) = {
	do(
		m.relaxfield(nn,scale=fsc);
		normalize(nn);
		print(i, " ", m.totalenergy(), " Nematic:", lf.total(m[1]), " Line tension:", lt.total(m[1], bnd), " Anchoring:", la.total(m[1], bnd));
	,{i,niter})
}

minimizeshape(niter) = {
	do(
		m.linesearch(scalelimit=0.05);
		normalize(nn);
		print(i, " ", m.totalenergy(), " Nematic:", lf.total(m[1]), " Line tension:", lt.total(m[1], bnd), " Anchoring:", la.total(m[1], bnd));
	,{i,niter})
}

// Regularization
regularize(a, energy, area, eaw, limit, n) = {
	// Evaluate and set the energy weight
	feq=energy.integrand(a[1]);
	farea=area.integrand(a[1]);
	fweight=feq/farea;
	eaw.setweight(fweight);

	// Fix the boundary
	bnd=a[1].selectboundary();
	a[1].fix(bnd);

	out=table(
		a.linesearch(leq, scale=limit/2, scalelimit=limit);
		reg=leq.total(a[1]);
		print(reg);
		a[1].equiangulate();
		reg
		,{i,n}
	);

	// Unfix the boundary
	a[1].unfix(bnd);

	out
}

// Adaptive refinement
adaptiverefine(a, mult, func) = {
	f = func.integrand(a[1]);
	threshold=mult*f.total()/length(f.tolist());
	g = f.map(function(x, x>threshold));
	p = a.select(g).changegrade({0,1});
	a.refine(p);
	a[1].equiangulate();
};


// Minimization
minimize(niter, nreg, fsc, reglimit) = {
	do(
		minimizeshape(1);
		minimizefield(10, fsc);
		if(nreg>0,
			regularize(m, lf, laa, leq, reglimit, nreg);
			minimizefield(10, fsc);
		);

		,{i,niter}
	);
}

/* ---------------------------------------------------------------
 * Main work
 * --------------------------------------------------------------- */

minimizefield(50, 0.25)

minimize(10, 0, 0.25, 0)

m.refine()

minimize(10, 1, 0.125, 0.001)

//adaptiverefine(m, 1.5, lf)

//minimize(10, 1, 0.125, 0.001)

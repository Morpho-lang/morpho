// cube.morpho
// Evolve a cube into a sphere

Nsteps=100
Nlevel=4

draweveryiteration=true

// Create the initial mesh
c=cuboid({-1/2,-1/2,-1/2}, {1/2,1/2,1/2})

// Initial state
//show(c)

//c.draw().tographics().export(stringjoin("cube",tostring(0),".eps"));

// Add energies
s=new(surfacetension)
c.addenergy(s)
v=new(enclosedvolume)
c.addconstraint(v)

c.relax(scale=0.1)

area={};
vol={};

// Loop over refinement levels
do(
    print("Iteration ", k);

    do(
	    c.relax(scale=0.1);
		//if(draweveryiteration,
			// c.draw().tographics().export(stringjoin("cube",tostring(k),"-",tostring(i),".eps"))
		//);
    , {i,1,Nsteps});

    // Extract quantitative date from the table
    print(c.totalenergy());

    if (k<Nlevel, c.refine(););

    ,{k,1,Nlevel});

//c[1].export("sphere3.mesh")

// Plot and export it
//plotlist(area).export("cubearea.eps")
//plotlist(vol).export("cubevol.eps")

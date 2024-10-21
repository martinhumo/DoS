import numpy as np
from ovito.io import import_file
from tqdm import tqdm
 
class Trajectory:
    def __init__(self, filename, attr, skip, fcv, fcx, f_stop=None, mask=None):
            """
            filename         : path to the trajectory file with sorted and format = id mol x y z fx fy fz vx vy vz
            attr             : particles attributes to be loaded in memory = coordinates(0), forces(1) or velocities(2)
            skip             : number of snapshots to be skipped between two configurations that are evaluated
                            (for example, if trajectory is 9000 steps long, and skip = 10, every tenth step
                            is evaluated, 900 steps in total; use skip = 1 to take every step of the MD)
            fcv              : conversion factor to have velocities  [m/s]
            fcx              : conversion factor to have positions in [m]
            f_stop           : last frame to be loaded. If f_stop = None, all frames are loaded.
            mask             : boolean array indicating which atoms to select in a single molecule
            """

            pipeline = import_file(filename) #import file
            self.n_atoms = pipeline.compute().particles.count #number of atoms
            self.n_steps_total = pipeline.source.num_frames

            self.skip = skip
            if f_stop == None:
                self.n_steps = self.n_steps_total // self.skip
            else:
                if f_stop > self.n_steps_total:
                    raise ValueError('f_stop > n_steps_total')
                else:
                    self.n_steps = f_stop // self.skip

            attributes = ('coordinates', 'forces', 'velocities')
            print('Are going to be loaded: particles {}'.format(attributes[attr]))
            print('Trajectory frames= ',self.n_steps_total)
            print('Frames to be loaded= ',self.n_steps)


            # Create a mask for the specified atom range
            if mask is not None:
                n_atoms_per_molecule = len(mask)
                n_molecules = self.n_atoms // n_atoms_per_molecule
                atom_mask = np.tile(mask, n_molecules)
            else:
                atom_mask = np.ones(self.n_atoms, dtype=bool)

            # Calculate the number of selected atoms
            n_selected_atoms = np.sum(atom_mask)
            self.n_atoms = n_selected_atoms

            self.coordinates = np.empty((self.n_steps, n_selected_atoms, 3))
            if attr == 0:
                self.boxsize = np.empty((self.n_steps, 3, 2))
            count = 0
            stop = self.n_steps

            print('--- Loading Trajectory ---')
            for step in tqdm(range(self.n_steps)):

                frame = step * self.skip 
                try:
                    if attr == 0:
                        self.coordinates[step] = pipeline.compute(frame).particles.positions[atom_mask] * fcx
                        self.boxsize[step,:,0] = pipeline.compute(frame).cell[:,3] * fcx
                        self.boxsize[step,:,1] = np.sum(pipeline.compute(frame).cell[:,:], axis=1) * fcx
                    elif attr == 1:
                        self.coordinates[step] = pipeline.compute(frame).particles.forces[atom_mask]
                    elif attr == 2:
                        self.coordinates[step] = pipeline.compute(frame).particles.velocities[atom_mask] * fcv
                except:
                    if count == 0:
                        stop = step
                        print( 'file broken in step: ',step * self.skip)
                    count += 1
                    break
            self.coordinates =  self.coordinates[:stop,:,:]
            if attr == 0:
                self.boxsize = self.boxsize[:stop,:,:]


    def Return_DOS_trn(self,tstep,Dstep,temp,m,nb):
        """This function return the translational (DOS). 
        1° Takes de power spectral density  of  molecule's  mass center velocities, 
        2° Sum it for x,y,z and whole system 
        3° weigh the sum with kb*T 
        Returns: freq [1/s] ; Dos_trn[tot,x,y,z] [s]  ; CMvs[x,y,z] [m/s]
        Needs:
        *self.coordinates: atoms velocities [a.u.]
        *tstep: simulation timestep [s]
        *Dstep: every few timestep each frame is recorded in the simulation []
        *temp: temperature [K]
        *m: atoms mass (sorted array for molecule)[kg]
        *nb: atoms per molecule []  
        """
        from scipy.constants import Boltzmann 
        kb = Boltzmann     #Boltzmann's constant [J/K]   
        nm = self.n_atoms//nb                             #molecules number
        Ts = tstep*Dstep*self.skip                        #sampling period
        mi = np.sum(m)                                    #molecule mass
        T = temp                                          #simulation temperature
        n = self.n_steps*4                                #number of points to do fft (if n>steps vector it will be zero padded) []

        CMvs = np.zeros((self.n_steps,nm, 3))   #molecule mass center velocities
        for step in range(self.n_steps):
            data_frame = np.array(self.coordinates[step])
            for mol in np.arange(1,nm+1,1):
                beadvs = data_frame[(mol-1)*nb:(mol*nb),:]    #beads velocitys in each particle              
                CMvs[step,mol-1,:] = np.average(beadvs, axis=0, weights=m)   #molecule mass center velocity
        
        fft  = np.zeros((n//2 +1,4)) #Sigle sided fft (only freq > 0 )
        for mol in range(nm):
            CMv = CMvs[:,mol,:]
            for i in range(3):   #(xyz)
                ffti = ( Ts  / (self.n_steps-1))*(np.abs(np.fft.rfft(CMv[:,i], n=n))**2)*mi 
                fft[:, i+1] += ffti
                fft[:, 0] += ffti   
        
        freq = np.fft.rfftfreq(n, d=Ts)
        DOS = ((2.)/(kb*T))*fft
        return DOS, freq, CMvs

    def Return_DOS_partition_trn(self,DOSt,freq,temp,m,nb,rho):
        """This function return the translational Dos_g_trn gas 
        and Dos_s_trn solid, Delta and fluidicity f. 
        Returns: c*Dos_trn [m] c*Dos_g_trn [m] ; c*Dos_s_trn [m] ; freq/c [1/m] ;
        f [] ; Delta []  
        Needs:
        *DOSt: Total density of states [s]
        *freq: frequencies [1/s] 
        *tstep: simulation timestep [s]
        *Dstep: every few timestep each frame is recorded []
        *temp: temperature [K]
        *m: atoms mass (sorted array for molecule)[kg]
        *nb: atoms per molecule []
        *rho: density in [kg/m3]
        """
        import scipy.optimize as opt
        from scipy.constants import c, Boltzmann #light speed in [m/s]
        kb = Boltzmann     #Boltzmann's constant [J/K]   
        nm = self.n_atoms//nb                             #particles number
        mi = np.sum(m)                                    #particle mass
        T = temp                                          #simulation temperature
 
        cDOSt = c*DOSt
        vfreq = freq/c
        #fluidicity f
        cdos0 = cDOSt[0] #gas-like difussive component
        Delta = (2*(cdos0/c)/(9*nm))*np.sqrt((np.pi*kb*T)/mi)*((rho/mi)**(1/3))*((6/np.pi)**(2/3))
 
        def y(f,Delta):
            return 2*(Delta**(-9/2))*(f**(15/2)) - 6*(Delta**(-3))*(f**5) - \
            (Delta**(-3/2))*(f**(7/2)) + 6*(Delta**(-3/2))*(f**(5/2)) + \
            2*f - 2

        f = opt.root_scalar(y, args=(Delta,), method='toms748', bracket=(0.,1.), rtol=1.0e-6, maxiter=int(1e6)).root #Busqueda de la raiz entre 0 y 1.

        cDOSg =  cdos0 / (1 + ((np.pi*cdos0*vfreq) /(6*f*nm))**2)       #DOS gas-like component
        cDOSs = cDOSt - cDOSg                  #DOS solid-like component

        return cDOSt, cDOSg, cDOSs, vfreq, f, Delta

    def Return_thermoproperties_trn(self,cDOSg,cDOSs,vfreq,T,m,f,Delta,V,nb,stat,weight_f=False):
        """Function that return Internal Energy, Entropy and Helmoltz Free Energy
        only for trn component.
        Returns: E/particle [J]; S/particle [J/K] ; A/particle [J]
        Needs:
        * c*Dos_g_trn: gas-like DOS [m] 
        * c*Dos_s_trn solid-like DOS[m] 
        * freq/c: wavenumber [1/m] 
        * T: temperature [K]
        * m: particle mass [kg]
        * f: fluidicity parameter []
        * Delta: normalized diffusivity []
        * V: Box volume [m3]
        * nb: beads per particle []
        * stat: quantum or classic solid-like weighting function ['q' or 'c']
        * weight_f: return weighting function (default: False)
        """
        import scipy.integrate as integrate
        from scipy.constants import c,h,Boltzmann #light speed in [m/s], #Planck's constant  [J*s]
        kb = Boltzmann     #Boltzmann's constant [J/K]   
        nm = self.n_atoms//nb                             #particles number
        mi = np.sum(m)                                    #particle mass

        #Energy
        B = 1/(kb*T)
        Bhv = B*h*(vfreq*c)
        if stat == 'c':
            Wes = 1                             #classical weight
        elif stat == 'q':
            Wes = Bhv*0.5 + Bhv / (np.exp(Bhv)-1)#Quantum weight
            Wes[0] = 1. 
        Es = integrate.simps(cDOSs*Wes,x=vfreq)

        Weg = 0.5
        Eg =  integrate.simps(cDOSg*Weg,x=vfreq)

        E = (Es+Eg)/B
        E /= nm
        

        #Entropy S
        if stat == 'c':   
            Wss = 1 - np.log(Bhv)                            #Classic weighting
            Wss[0] = 1 - np.log(1.e-323) 
        elif stat == 'q':
            Wss = Bhv / (np.exp(Bhv)-1) - np.log(1 - np.exp(-Bhv)) #quatum weighting
            Wss[0] = 0. 
        Ss =  integrate.simps(cDOSs*Wss,x=vfreq)

        y = (f**(5./2.)) / (Delta**(3./2.))
        z = (1 + y + y**2 - y**3 ) / ((1 - y)**3)
        Wsg =(1./3.)*((5./2.) + np.log(((2.*np.pi*mi*kb*T)/ h**2)**(3./2.) * ((V*z)/(f*nm))) + y*(3*y-4)/(1-y)**2 )
        Sg =  integrate.simps(cDOSg*Wsg,x=vfreq)

        S =kb*(Ss+Sg)
        S /= nm
     

        #Helmoltz Free Energy
        if stat == 'c':                              #Classic weighting
            Was = np.log(Bhv)
            Was[0] = np.log(1.e-323)
        elif stat == 'q':
            Was = np.log((1-np.exp(-Bhv))/(np.exp(-Bhv/2.))) #quatum weighting
            Was[0] = np.log(1.e-323)
        As = integrate.simps(cDOSs*Was,x=vfreq)

        Wag = Weg - Wsg
        Ag =  integrate.simps(cDOSg*Wag,x=vfreq)

        A =  (As+Ag)/B
        A /= nm

        return (E, S, A, Wes, Weg, Wss, Wsg, Was, Wag) if weight_f else (E, S, A)

    def Return_DOS_rot(self,rposis,tstep,Dstep,temp,m,nb):
        """This function return the rotational (DOS).
        1° calculates the molecule mass translational velocity;
        2° Calculates the molecule angular velocity around principal axis;
        3° Takes de power spectral density  of  molecule  mass center angular velocities,
        4° Sum it for x,y,z and whole system
        5° weigh the sum with kb*T
        Returns: freq [1/s] ; Dos_rot[tot,a,b,c] [s] ; <I> [Kgm²] ; CMws[a,b,c] [rad/s] ; Imom[a,b,c] [kgm²]
        Needs:
        *self.coordinates: velocities [m/s]
        *rposi: distance [m]
        *tstep: simulation timestep [s]
        *Dstep: every few timestep each frame is recorded in the simulation []
        *temp: temperature [K]
        *m: atom mass (sorted array for molecule)[kg]
        *nb: atoms per molecule []

        References:
        (1)Application of the Eckart frame to soft matter: Rotation of star polymers under shear flow [2017]
        (2)Master thesis 2PT bernhartdt [2016]
        """
        from scipy.constants import Boltzmann 
        kb = Boltzmann     #Boltzmann's constant [J/K]   
        nm = self.n_atoms//nb                             #molecules number
        Ts = tstep*Dstep*self.skip                        #sampling period
        T = temp                                          #simulation temperature
        n = self.n_steps*4                                #number of points to do fft (if n>steps vector it will be zero padded) []

        CMvs = np.zeros((self.n_steps,nm, 3))   #molecule mass center velocities
        for step in range(self.n_steps):
            data_frame = np.array(self.coordinates[step])
            for mol in np.arange(1,nm+1,1):
                beadvs = data_frame[(mol-1)*nb:(mol*nb),:]    #beads velocities in each molecule
                CMvs[step,mol-1,:] = np.average(beadvs, axis=0, weights=m)   #molecule mass center velocity



        CMws = np.zeros((self.n_steps,nm, 3))   #molecule angular velocity
        Imom = np.zeros((self.n_steps,nm, 3))
        for mol in np.arange(1,nm+1,1):
            for step in range(self.n_steps):

                data_rposi = rposis[step,(mol-1)*nb:(mol*nb),:]  #See eq 2 and 3 in: (1)
                data_vicm = self.coordinates[step,(mol-1)*nb:(mol*nb),:] - CMvs[step,mol-1,:]

                L = np.sum([np.cross(data_rposi[i,:],data_vicm[i,:]*m[i]) for i in range(nb) ],axis=0) #angular momentum
                J = np.sum( [ m[i]*(np.dot(data_rposi[i,:],data_rposi[i,:])*np.identity(3) \
                            - np.tensordot(data_rposi[i,:],data_rposi[i,:],axes=0)) for i in range(nb) ],axis=0) #inertia matrix

                eival, eivec = np.linalg.eig(J) #Principal intertia  moments and axis
                idx = eival.argsort()[::-1] #highest to lowest
                eival = eival[idx]
                eivec = eivec[:,idx]
                Imom[step,mol-1,:] = eival
                #lines to avoid eigvect +1 o -1 directions
                if step == 0:
                    for i in range(3):
                        CMws[step,mol-1,i] = (1./eival[i])*np.dot(L,eivec[:,i])*np.sqrt(eival[i]) #See eq. 3.3 Master thesis 2PT bernhartdt
                    eivecprev = eivec
                else:
                    for i in range(3):
                        test = np.dot(eivec[:,i],eivecprev[:,i])
                        if test < 0.:
                            eivec[:,i] *= -1. #if eivec point in inverse direction from previous step
                        CMws[step,mol-1,i] = (1./eival[i])*np.dot(L,eivec[:,i])*np.sqrt(eival[i]) #srqt(I_jl)*w_jl
                    eivecprev = eivec

        fft  = np.zeros((n//2 + 1, 4)) #Sigle sided fft (only freq > 0 )
        for mol in range(nm):
            CMw = CMws[:,mol,:]
            for i in range(3):   #(xyz)
                ffti = ( Ts  / (self.n_steps-1))*(np.abs(np.fft.rfft(CMw[:,i], n=n))**2)
                fft[:, i+1] += ffti
                fft[:, 0] += ffti

        I = np.mean(Imom[:,:,:],axis=(0,1))
        freq = np.fft.rfftfreq(n, d=Ts)
        DOS = ((2.)/(kb*T))*fft
        return DOS, freq, I, CMws, Imom



    def compute_rposi(self,m,nb):
        """This function calculates atom vector position to
        molecule mass center (r_i - r_cm).
        Needs:
        *self.coordinates: atoms positions [a.u.]
        *m: atom mass (sorted array for molecule)[kg]
        *nb: atoms per molecule []
        """
        nm = self.n_atoms//nb #molecules number

        rposis = np.zeros((self.n_steps, self.n_atoms, 3))
        for step in range(self.n_steps):
            data_box    = np.array(self.boxsize[step])#Step box size
            A = (data_box[0,1] - data_box[0,0])*0.5
            B = (data_box[1,1] - data_box[1,0])*0.5
            C = (data_box[2,1] - data_box[2,0])*0.5
            box = np.array([A,B,C])

            data_beads = np.array(self.coordinates[step,:,:])
            for mol in np.arange(1,nm+1,1):
                molecule = data_beads[(mol-1)*nb:(mol*nb),:]

                moleculeu = np.zeros(molecule.shape) #molecule with unwrapped cordinates
                moleculeu[0,:] = molecule[0,:]
                for i in range(nb-1):
                    dist = (molecule[i+1]-moleculeu[i])
                    for xyz in (0,1,2):
                        if dist[xyz]>box[xyz]:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] - box[xyz]*2.
                        elif dist[xyz]<=(-box[xyz]):
                            moleculeu[i+1,xyz] = molecule[i+1,xyz] + box[xyz]*2.
                        else:
                            moleculeu[i+1,xyz] = molecule[i+1,xyz]
                CM = np.average(moleculeu, axis=0, weights=m)
                rposis[step,(mol-1)*nb:(mol*nb),:] = moleculeu[:,:] - CM[:]
        return rposis


    def Return_DOS_partition_rot(self,DOSt,freq,temp,m,nb,rho):
        """This function return the translational Dos_g_trn gas
        and Dos_s_trn solid, Delta and fluidicity f.
        Returns: c*Dos_trn [m] c*Dos_g_trn [m] ; c*Dos_s_trn [m] ; freq/c [1/m] ;
        f [] ; Delta []
        Needs:
        *DOSt: Total density of states [s]
        *freq: frequencies [1/s]
        *temp: temperature [K]
        *m: atoms mass (sorted array for molecule)[kg]
        *nb: atoms per molecule []
        *rho: density in [kg/m3]
        """
        import scipy.optimize as opt
        from scipy.constants import c, Boltzmann #light speed in [m/s]
        kb = Boltzmann     #Boltzmann's constant [J/K]   
        nm = self.n_atoms//nb                             #particles number
        mi = np.sum(m)                                    #particle mass
        T = temp                                          #simulation temperature

        cDOSt = c*DOSt
        vfreq = freq/c
        #fluidicity f
        cdos0 = cDOSt[0] #gas-like difussive component
        Delta = (2*(cdos0/c)/(9*nm))*np.sqrt((np.pi*kb*T)/mi)*((rho/mi)**(1/3))*((6/np.pi)**(2/3)) 

        def y(f,Delta):
            return 2*(Delta**(-9/2))*(f**(15/2)) - 6*(Delta**(-3))*(f**5) - \
            (Delta**(-3/2))*(f**(7/2)) + 6*(Delta**(-3/2))*(f**(5/2)) + \
            2*f - 2

        f = opt.root_scalar(y, args=(Delta,), method='toms748', bracket=(0.,1.), rtol=1.0e-6, maxiter=int(1e6)).root #Busqueda de la raiz entre 0 y 1.

        cDOSg =  cdos0 / (1 + ((np.pi*cdos0*vfreq) /(6*f*nm))**2)       #DOS gas-like component
        cDOSs = cDOSt - cDOSg                  #DOS solid-like component

        return cDOSt, cDOSg, cDOSs, vfreq, f, Delta

    def Return_thermoproperties_rot(self,cDOSg,cDOSs,vfreq,T,I,nb,stat,sigma,weight_f=False):
        """Function that return Internal Energy, Entropy and Helmoltz Free Energy
        only for rotational component.
        Returns: E/particle [J]; S/particle [J/K] ; A/particle [J]
        Needs:
        * c*Dos_g_rot: gas-like DOS [m]
        * c*Dos_s_rot: solid-like DOS[m]
        * freq/c: wavenumber [1/m]
        * T: temperature [K]
        * I: Principal Moments Of Inertia (sorted array for molecule)[Kgm²]
        * nb: beads per particle []
        * stat: quantum or classic solid-like weighting function ['q' or 'c']
        * sigma: Rotational symmetry []
        * weight_f: return weighting function (default: False)
        """
        import scipy.integrate as integrate
        from scipy.constants import c,h,Boltzmann #light speed in [m/s], #Planck's constant  [J*s]
        kb = Boltzmann     #Boltzmann's constant [J/K]   
        nm = self.n_atoms//nb                             #particles number

        #Energy
        B = 1/(kb*T)
        Bhv = B*h*(vfreq*c)
        if stat == 'c':
            Wes = 1                             #classical weight
        elif stat == 'q':
            Wes = Bhv*0.5 + Bhv / (np.exp(Bhv)-1)#Quantum weight
            Wes[0] = 1.
        Es = integrate.simps(cDOSs*Wes,x=vfreq)

        Weg = 0.5
        Eg =  integrate.simps(cDOSg*Weg,x=vfreq)

        E =  (Es+Eg)/B
        E /= nm

        #Entropy S
        if stat == 'c':
            Wss = 1 - np.log(Bhv)                            #Classic weighting
            Wss[0] = 1 - np.log(1.e-323)
        elif stat == 'q':
            Wss = Bhv / (np.exp(Bhv)-1) - np.log(1 - np.exp(-Bhv)) #quatum weighting
            Wss[0] = 0.
        Ss =  integrate.simps(cDOSs*Wss,x=vfreq)

        tit= np.array([ (h**2) / (8.*np.pi*I[i]*kb) for i in range(3)])
        Wsg = (1./3.)*np.log( ( ((np.pi**0.5)*np.exp(3/2)) / sigma ) *(T**3 / (tit[0]*tit[1]*tit[2])**0.5)) # Eq. 25b Lin2010
        Sg =  integrate.simps(cDOSg*Wsg,x=vfreq)

        S =kb*(Ss+Sg)
        S /= nm

        #Helmoltz Free Energy
        if stat == 'c':                              #Classic weighting
            Was = np.log(Bhv)
            Was[0] = np.log(1.e-323)
        elif stat == 'q':
            Was = np.log((1-np.exp(-Bhv))/(np.exp(-Bhv/2.))) #quatum weighting
            Was[0] = np.log(1.e-323)
        As = integrate.simps(cDOSs*Was,x=vfreq)

        Wag = Weg - Wsg
        Ag =  integrate.simps(cDOSg*Wag,x=vfreq)

        A =  (As+Ag)/B
        A /= nm

        return (E, S, A, Wes, Weg, Wss, Wsg, Was, Wag) if weight_f else (E, S, A)

    def Return_DOS_vib(self,rposis,tstep,Dstep,temp,m,nb):
        """This function return the vibrational (DOS).
        1° calculates the molecule mass translational velocity;
        2° Calculates the molecule angular velocity around principal axis;
        3° Calculates atoms vibrational velocities.
        4° Takes de power spectral density  of  atoms  mass center vibrational velocities,
        5° Sum it for x,y,z and whole system
        6° weigh the sum with kb*T
        Returns: freq [1/s] ; Dos_vib[tot,x,y,z] [s]
        Needs:
        *self.coordinates: velocities [a.u.]
        *rposi: distance to mass center [m]
        *tstep: simulation timestep [s]
        *Dstep: every few timestep each frame is recorded in the simulation []
        *temp: temperature [K]
        *m: atom mass (sorted array for molecule)[kg]
        *nb: atoms per molecule []

        References:
        (1)Application of the Eckart frame to soft matter: Rotation of star polymers under shear flow [2017]
        (2)Master thesis 2PT bernhartdt [2016]
        """
        from scipy.constants import Boltzmann 
        kb = Boltzmann     #Boltzmann's constant [J/K]   
        nm = self.n_atoms//nb                             #molecules number
        Ts = tstep*float(Dstep)*float(self.skip)                        #sampling period
        T = temp                                          #simulation temperature
        n = self.n_steps*4                                #number of points to do fft (if n>steps vector it will be zero padded) []

        CMvs = np.zeros((self.n_steps,nm, 3))   #molecule mass center velocities
        for step in range(self.n_steps):
            data_frame = np.array(self.coordinates[step])
            for mol in np.arange(1,nm+1,1):
                beadvs = data_frame[(mol-1)*nb:(mol*nb),:]    #beads velocities in each molecule
                CMvs[step,mol-1,:] = np.average(beadvs, axis=0, weights=m)   #molecule mass center velocity



        CMws  = np.zeros((self.n_steps,nm, 3))   #molecule angular velocity
        vibvs = np.zeros((self.n_steps,self.n_atoms, 3)) # atoms vibrational velocities
        for mol in np.arange(1,nm+1,1):
            for step in range(self.n_steps):

                data_rposi = rposis[step,(mol-1)*nb:(mol*nb),:]  #See eq 2 and 3 in: (1)
                data_vicm = self.coordinates[step,(mol-1)*nb:(mol*nb),:] - CMvs[step,mol-1,:]

                L = np.sum([np.cross(data_rposi[i,:],data_vicm[i,:]*m[i]) for i in range(nb) ],axis=0) #angular momentum
                J = np.sum( [ m[i]*(np.dot(data_rposi[i,:],data_rposi[i,:])*np.identity(3) \
                            - np.tensordot(data_rposi[i,:],data_rposi[i,:],axes=0)) for i in range(nb) ],axis=0) #inertia matrix

                CMws[step,mol-1,:] = np.dot(np.linalg.inv(J),L)
                vibvs[step,(mol-1)*nb:(mol*nb),:] = (np.array(self.coordinates[step,(mol-1)*nb:(mol*nb),:]) - \
                                                     (CMvs[step,mol-1,:] + np.cross(CMws[step,mol-1,:], rposis[step,(mol-1)*nb:(mol*nb),:])))



        fft  = np.zeros((n//2 +1, 4)) #Sigle sided fft (only freq > 0 )
        m_system = np.tile(m, nm) #Array with system masses
        for atom in range(self.n_atoms):
            vibv = vibvs[:,atom,:]
            for i in range(3):   #(xyz)
                ffti = ( Ts  / (self.n_steps-1))*(np.abs(np.fft.rfft(vibv[:,i], n=n))**2)*m_system[atom]
                fft[:,i+1] += ffti
                fft[:,0] += ffti

        freq = np.fft.rfftfreq(n, d=Ts)
        DOS = ((2.)/(kb*T))*fft
        return DOS, freq



    def Return_DOS_partition_vib(self,DOSt,freq):
        """This function return the vibrational Dos_s_vib solid,
        fluidicity f = 0 .
        Returns: c*Dos_vib [m]; c*Dos_s_vib [m] ; freq/c [1/m] ;
        Needs:
        *DOSt: Total density of states [s]
        *freq: frequencies [1/s]
        """
        import scipy.optimize as opt
        from scipy.constants import c #light speed in [m/s]

        cDOSt = c*DOSt
        vfreq = freq/c
        cDOSs = cDOSt                   #DOS solid-like component

        return cDOSt, cDOSs, vfreq

    def Return_thermoproperties_vib(self,cDOSs,vfreq,T,nb,stat,weight_f=False):
        """Function that return Internal Energy, Entropy and Helmoltz Free Energy
        only for rotational component.
        Returns: E/particle [J]; S/particle [J/K] ; A/particle [J]
        Needs:
        * c*Dos_s_trn solid-like DOS[m]
        * freq/c: wavenumber [1/m]
        * T: temperature [K]
        * nb: beads per particle []
        * stat: quantum or classic solid-like weighting function ['q' or 'c']
        * weight_f: return weighting function (default: False)
        """
        import scipy.integrate as integrate
        from scipy.constants import c,h,Boltzmann #light speed in [m/s], #Planck's constant  [J*s]
        kb = Boltzmann     #Boltzmann's constant [J/K]                       
        nm = self.n_atoms//nb                             #particles number

        #Energy
        B = 1/(kb*T)
        Bhv = B*h*(vfreq*c)
        if stat == 'c':
            Wes = 1                             #classical weight
        elif stat == 'q':
            Wes = Bhv*0.5 + Bhv / (np.exp(Bhv)-1)#Quantum weight
            Wes[0] = 1.
        Es = integrate.simps(cDOSs*Wes,x=vfreq)

        E = (Es)/B
        E /= nm

        #Entropy S
        if stat == 'c':
            Wss = 1 - np.log(Bhv)                            #Classic weighting
            Wss[0] = 1 - np.log(1.e-323)
        elif stat == 'q':
            Wss = Bhv / (np.exp(Bhv)-1) - np.log(1 - np.exp(-Bhv)) #quatum weighting
            Wss[0] = 0.
        Ss =  integrate.simps(cDOSs*Wss,x=vfreq)

        S =kb*(Ss)
        S /= nm

        #Helmoltz Free Energy
        if stat == 'c':                              #Classic weighting
            Was = np.log(Bhv)
            Was[0] = np.log(1.e-323)
        elif stat == 'q':
            Was = np.log((1-np.exp(-Bhv))/(np.exp(-Bhv/2.))) #quatum weighting
            Was[0] = np.log(1.e-323)
        As = integrate.simps(cDOSs*Was,x=vfreq)

        A =  (As)/B
        A /= nm

        return (E, S, A, Wes, Wss, Was) if weight_f else (E, S, A)
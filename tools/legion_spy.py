#!/usr/bin/python

import subprocess
import pygame, sys, os, shutil
import string
from pygame.locals import *
from getopt import getopt

from parser import parse_log_file 

temp_dir = ".spy/"

class ImageWrapper(object):
    def __init__(self,name,file_name,fontObj=None,blackColor=None,scale=10):
        self.name = name
        self.surface = pygame.image.load(file_name) 
        if fontObj <> None:
            self.origin = (0,0)
            self.file_name = file_name
            self.prefix = string.split(file_name,'.png')[0]
            self.msgSurface = fontObj.render(self.name,False,blackColor)
            self.msgRect = self.msgSurface.get_rect()
            self.larger = None
            self.smaller = None
            self.zoom = 100 # 100 percent
            self.scale = scale # Scale by 10% each time

    def move_left(self,delta):
        self.origin = (self.origin[0]+delta,self.origin[1])

    def move_down(self,delta):
        self.origin = (self.origin[0],self.origin[1]-delta)

    def move_up(self,delta):
        self.origin = (self.origin[0],self.origin[1]+delta)

    def move_right(self,delta):
        self.origin = (self.origin[0]-delta,self.origin[1])

    def reset(self):
        self.origin = (0,0)

    def dump_file(self):
        subprocess.call(['cp '+self.file_name+' .'],shell=True)

    def display(self,target):
        target.blit(self.surface, (self.origin[0],self.origin[1]+self.msgRect.height))
        target.blit(self.msgSurface,(0,0))

    def zoom_in(self):
        # Check to see if we already have a zoomed in version
        if (self.larger <> None):
            self.larger.origin = self.origin
            return self.larger
        zoom_image_name = self.prefix + '_' + str(self.zoom+self.scale) + '.png'
        # Do the conversion
        subprocess.call(['convert '+self.prefix+'.png -scale '+str(self.zoom+self.scale)+'% '+zoom_image_name],shell=True)
        # Create the new image
        self.larger = ImageWrapper(self.name,zoom_image_name)
        self.larger.origin = self.origin
        self.larger.prefix = self.prefix
        self.larger.msgSurface = self.msgSurface
        self.larger.msgRect = self.msgRect
        self.larger.smaller = self
        self.larger.larger = None
        self.larger.zoom = self.zoom + self.scale
        self.larger.scale = self.scale
        return self.larger

    def zoom_out(self):
        if (self.smaller <> None):
            self.smaller.origin = self.origin
            return self.smaller
        # Dont' zoom past 10%
        if (self.zoom == self.scale):
            return self
        zoom_image_name = self.prefix + '_' + str(self.zoom-self.scale) + '.png'
        # Do the conversion
        subprocess.call(['convert '+self.prefix+'.png -scale '+str(self.zoom-self.scale)+'% '+zoom_image_name],shell=True)
        # Create the new image
        self.smaller = ImageWrapper(self.name,zoom_image_name)
        self.smaller.origin = self.origin
        self.smaller.prefix = self.prefix
        self.smaller.msgSurface = self.msgSurface
        self.smaller.msgRect = self.msgRect
        self.smaller.larger = self
        self.smaller.smaller = None
        self.smaller.zoom = self.zoom - self.scale
        self.smaller.scale = self.scale
        return self.smaller
        

def usage():
    print "Usage: "+ sys.argv[0] +" [-h (height in pixels)] [-w (width in pixels)] [-d (delta move size in pixels)] [-z (zoom percentage)] file_name"
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        usage()

    opts, args = getopt(sys.argv[1:],'d:h:w:z:')
    opts = dict(opts)
    if len(args) <> 1: 
        usage()

    height = int(opts.get('-h',750))
    width = int(opts.get('-w',1500))
    delta = int(opts.get('-d',10))
    scale = int(opts.get('-z',10))

    file_name = args[0]
    print 'Loading log file '+file_name+'...'
    log = parse_log_file(file_name)

    pygame.init()
    fpsClock = pygame.time.Clock()

    showMode = 0
    numModes = 3
    surface = pygame.display.set_mode((width,height))
    pygame.display.set_caption('Legion Spy')

    redColor = pygame.Color(255,0,0)
    greenColor = pygame.Color(0,255,0)
    blueColor = pygame.Color(0,0,255)
    whiteColor = pygame.Color(255,255,255)
    blackColor = pygame.Color(0,0,0)

    fontObj = pygame.font.Font(pygame.font.get_default_font(),32)

    # Make a temporary directory
    print 'Generating images...'
    tree_files = log.print_trees(temp_dir)
    tree_images = list()
    for t,pair in sorted(tree_files.iteritems()):
        tree_images.append(ImageWrapper(pair[1],pair[0],fontObj,blackColor,scale))

    ctx_files = log.print_contexts(temp_dir)
    ctx_images = list()
    for ctx_id,pair in sorted(ctx_files.iteritems()):
        ctx_images.append(ImageWrapper(pair[1],pair[0],fontObj,blackColor,scale)) 

    event_file = log.print_event_graph(temp_dir)
    event_image = ImageWrapper('Event Image',event_file,fontObj,blackColor,scale)

    currentImage = tree_images[0]
    currentTree = 0
    currentCtx  = 0

    print 'Initializing display...'

    movingLeft = False
    movingRight = False
    movingDown = False
    movingUp = False
        
    while True:
        justSet = False
        unSet = False
        # Handle all the events
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYUP:
                if not justSet:
                    if event.key == K_h:
                        movingLeft = False
                    elif event.key == K_j:
                        movingDown = False
                    elif event.key == K_k:
                        movingUp = False
                    elif event.key == K_l:
                        movingRight = False
                else:
                    unSet = True
            elif event.type == KEYDOWN:
                if event.key == K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == K_h: # move right 
                    movingRight = True
                    justSet = True
                elif event.key == K_j: # move up 
                    movingUp = True    
                    justSet = True
                elif event.key == K_k: # move down 
                    movingDown = True 
                    justSet = True
                elif event.key == K_l: # move left 
                    movingLeft = True
                    justSet = True
                elif event.key == K_i: # zoom in
                    currentImage = currentImage.zoom_in()
                    if showMode == 0:
                        tree_images[currentTree] = currentImage
                    elif showMode == 1:
                        ctx_images[currentCtx] = currentImage
                    else:
                        event_image = currentImage
                elif event.key == K_o: # zoom out
                    currentImage = currentImage.zoom_out()
                    if showMode == 0:
                        tree_images[currentTree] = currentImage
                    elif showMode == 1:
                        ctx_images[currentCtx] = currentImage
                    else:
                        event_image = currentImage
                elif event.key == K_r: # reset
                    currentImage.reset()
                elif event.key == K_LEFT: # next picture 
                    if showMode == 0:
                        if currentTree == 0:
                            currentTree = len(tree_images)-1
                        else:
                            currentTree = currentTree-1
                        currentImage = tree_images[currentTree]
                    elif showMode == 1:
                        if currentCtx == 0:
                            currentCtx = len(ctx_images)-1
                        else:
                            currentCtx = currentCtx-1
                        currentImage = ctx_images[currentCtx]
                elif event.key == K_RIGHT: # previous picture
                    if showMode == 0:
                        currentTree = ((currentTree+1) % len(tree_images))
                        currentImage = tree_images[currentTree]
                    elif showMode == 1:
                        currentCtx = ((currentCtx+1) % len(ctx_images))
                        currentImage = ctx_images[currentCtx]
                elif event.key == K_DOWN:
                    showMode = (showMode + 1) % numModes
                    if showMode == 0:
                        currentImage = tree_images[currentTree]
                    elif showMode == 1:
                        currentImage = ctx_images[currentCtx]
                    else:
                        currentImage = event_image
                elif event.key == K_UP:
                    if showMode == 0: 
                        showMode = numModes-1
                    else:
                        showMode = showMode-1
                    if showMode == 0:
                        currentImage = tree_images[currentTree]
                    elif showMode == 1:
                        currentImage = ctx_images[currentCtx]
                    else:
                        currentImage = event_image
                elif event.key == K_p: #print the current image
                    currentImage.dump_file()
                    

        # Handle all moves
        if movingLeft:
            currentImage.move_left(delta)
        if movingDown: 
            currentImage.move_down(delta)
        if movingUp:
            currentImage.move_up(delta)
        if movingRight:
            currentImage.move_right(delta)

        if unSet:
            movingLeft = False
            movingDown = False
            movingUp = False
            movingRight = False

        # Render the scene
        surface.fill(whiteColor)
        currentImage.display(surface)

        pygame.display.update()
        fpsClock.tick(30)


if __name__ == "__main__":
    try:
        assert pygame.image.get_extended()
        os.mkdir(temp_dir)
        main()
        shutil.rmtree(temp_dir)
    except:
        # Remove the directory we created
        shutil.rmtree(temp_dir)
        raise

